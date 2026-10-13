//! A codec package's encoder and decoder as nodes of a network, the way
//! `codec.rs` runs either alone: one instance, called in order, its frames
//! cut as that run cuts them. Which half runs is what its pad carries: raw
//! frames are encoded, coded packets decoded.
//!
//! A module of `ffrwd:av` alone reads and writes bytes, so a picture held on
//! the GPU is read back once on its way into the encoder, made the format the
//! encoder reads on the device first where it is held in another; a decoded
//! picture is handed on in host memory, and a reader holding its frames on
//! the GPU uploads it. A module that also exports `ffrwd:gpu`'s `encoder` or
//! `decoder` is handed the planes as textures instead, and hands its decoded
//! planes back the same way, wherever the pictures are held on the GPU on
//! that side: nothing crosses host memory but the packets.

use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use ffrwd_wasm::nut;
use ffrwd_wasm_runtime::node::{
    Emission, Emitted, FrameData, Node, NodeShape, OutFrame, Payload, PortKind, StreamFormat, Tick,
};
use ffrwd_wasm_runtime::runtime::{
    self, CodedStream, Decoder, Encoder, Format, Media, Packet, RawFrame, StreamInfo, TimeBase,
};

use crate::av_nodes;
use crate::codec::{coded_header, coded_input, frame_ticks, raw_header, Framer};

/// The encoder of a codec package: raw frames in, its packets out.
pub struct EncoderNode {
    name: String,
    shape: NodeShape,
    encoder: Encoder,
    coded: CodedStream,
    header: nut::Stream,
    framer: Framer,
    /// The format the encoder reads, where the frames arrive held on the GPU
    /// in another, and the colour its YUV is written in: each is made it
    /// there before it is read back, in the colour the coded stream says,
    /// which is what a decoder reads it back in.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    convert: Option<(&'static str, av_nodes::Yuv)>,
    /// Where the frames arrive held on the GPU and the module reads
    /// textures: the format and colour it reads them in, and the picture
    /// not yet handed over.
    #[cfg(feature = "gpu")]
    textures: Option<Textures>,
}

/// Pictures on their way into an encoder that reads textures, made the
/// format it reads on the device, each handed over once its duration is
/// known: at once where the stream's rate settles it, else when the next
/// one arrives.
#[cfg(feature = "gpu")]
struct Textures {
    /// The format the frames arrive in, which one in host memory is put on
    /// the device as.
    arriving: &'static str,
    target: ffrwd_wasm_runtime::gpu::frames::Target,
    rate_ticks: Option<i64>,
    held: Option<(i64, Arc<ffrwd_wasm_runtime::gpu::frames::GpuFrame>)>,
}

#[cfg(feature = "gpu")]
impl Textures {
    /// A frame off the wire, made the format the encoder reads, and the
    /// pictures it completes. A frame in host memory is put on the device
    /// first.
    fn push(
        &mut self,
        pts: i64,
        data: &FrameData,
        out: &mut Vec<runtime::GpuPicture>,
    ) -> Result<()> {
        use ffrwd_wasm_runtime::gpu::frames::GpuFrame;
        let frame = match data {
            FrameData::Gpu(frame) => Arc::clone(frame),
            FrameData::Host(bytes) => GpuFrame::upload(
                self.arriving,
                self.target.width,
                self.target.height,
                Arc::clone(bytes),
            )?,
        };
        let frame = frame.converted(self.target)?;
        if let Some(ticks) = self.rate_ticks {
            out.push(runtime::GpuPicture {
                pts,
                duration: Some(ticks),
                frame,
            });
            return Ok(());
        }
        if let Some((held_pts, held)) = self.held.take() {
            out.push(runtime::GpuPicture {
                pts: held_pts,
                duration: Some(pts - held_pts).filter(|d| *d > 0),
                frame: held,
            });
        }
        self.held = Some((pts, frame));
        Ok(())
    }

    /// The last picture, lasting what the rate says.
    fn finish(&mut self, out: &mut Vec<runtime::GpuPicture>) {
        if let Some((pts, frame)) = self.held.take() {
            out.push(runtime::GpuPicture {
                pts,
                duration: self.rate_ticks,
                frame,
            });
        }
    }
}

impl EncoderNode {
    /// Opens the encoder of `path` on the frames `raw` describes, in `base`
    /// at `frame_rate`, under `header`. `pix_fmt` is the format the query
    /// settled for its pictures, and `on_gpu` says they arrive held on the
    /// GPU, where they are made that format, and handed over as textures
    /// where the module reads them so.
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        path: &str,
        params: &str,
        raw: &StreamFormat,
        info: &StreamInfo,
        base: TimeBase,
        frame_rate: Option<(u64, u64)>,
        header: &nut::Stream,
        pix_fmt: Option<&str>,
        on_gpu: bool,
    ) -> Result<EncoderNode> {
        let mut encoder = Encoder::load(path).with_context(|| format!("opening module {path}"))?;
        let name = encoder.name().to_string();
        let (media, kind, convert) = match raw {
            StreamFormat::Video(video) => {
                let wanted = pix_fmt.unwrap_or(video.pix_fmt);
                let takes = &encoder.described().meta.pixel_formats;
                let wanted = if pix_fmt.is_none() && !takes.iter().any(|f| f == wanted) {
                    takes.first().map_or(wanted, String::as_str)
                } else {
                    wanted
                };
                let mut video = *video;
                let convert = if wanted == video.pix_fmt {
                    None
                } else if on_gpu {
                    let to = ffrwd_wasm_runtime::av::WIRE_PIX_FMTS
                        .iter()
                        .copied()
                        .find(|f| *f == wanted)
                        .ok_or_else(|| {
                            anyhow!("{name} reads {wanted}, which the wire does not carry")
                        })?;
                    video.pix_fmt = to;
                    video.frame_len = crate::frame_len_for(to, video.width, video.height)?;
                    Some((to, av_nodes::yuv_of(video.color.as_ref())))
                } else {
                    bail!(
                        "{name} reads {wanted}, and its frames arrive in {} in host memory; \
                         only a picture held on the GPU is made another format on its way in",
                        video.pix_fmt
                    );
                };
                (Media::Video(video), PortKind::Video, convert)
            }
            StreamFormat::Audio(audio) => (Media::Audio(*audio), PortKind::Audio, None),
            other => bail!(
                "{name} is an encoder and reads raw frames; its pad carries {}",
                other.kind().name()
            ),
        };
        let format = Format {
            media,
            time_base: base,
        };
        let rate = frame_rate
            .and_then(|(num, den)| Some((i32::try_from(num).ok()?, i32::try_from(den).ok()?)));
        let coded = encoder.init(&format, info, rate, params)?;
        let described = encoder.described().clone();
        let color = match &format.media {
            Media::Video(video) => video.color,
            Media::Audio(_) => None,
        };
        let header = coded_header(
            &name,
            &described.fourcc,
            described.decode_delay,
            &coded,
            header,
            color.as_ref(),
        )?;
        let framer = Framer::new(format, frame_rate, described.frame_samples);
        #[cfg(feature = "gpu")]
        let textures = match format.media {
            Media::Video(video) if on_gpu && encoder.on_gpu() => {
                let arriving = match raw {
                    StreamFormat::Video(arriving) => arriving.pix_fmt,
                    _ => video.pix_fmt,
                };
                Some(textures_for(&name, arriving, &video, frame_rate, base)?)
            }
            _ => None,
        };
        Ok(EncoderNode {
            shape: av_nodes::shape(kind, PortKind::Packets, coded.time_base, 0.0),
            name,
            encoder,
            coded,
            header,
            framer,
            convert,
            #[cfg(feature = "gpu")]
            textures,
        })
    }

    pub fn coded(&self) -> &CodedStream {
        &self.coded
    }

    pub fn header(&self) -> &nut::Stream {
        &self.header
    }

    /// How deep its packets reorder, which the coded header carries.
    pub fn decode_delay(&self) -> u32 {
        self.encoder.described().decode_delay
    }

    /// A frame's bytes in the format the encoder reads.
    fn bytes(&self, data: &FrameData) -> Result<Vec<u8>> {
        #[cfg(feature = "gpu")]
        if let (Some((to, yuv)), FrameData::Gpu(frame)) = (self.convert, data) {
            use ffrwd_wasm_runtime::gpu::frames::{Layout, Target};
            let layout = Layout::of(to).expect("a wire format the device holds");
            let target = Target::new(layout, frame.width(), frame.height(), yuv);
            let made = frame
                .converted(target)
                .with_context(|| format!("{}: making a frame {to} on the device", self.name))?;
            return Ok(made.bytes()?.as_ref().clone());
        }
        Ok(data.bytes()?.as_ref().clone())
    }
}

impl Node for EncoderNode {
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
        Some(EncoderNode::decode_delay(self))
    }

    fn set_params(&mut self, _params: &str) -> Result<()> {
        bail!("{} takes no params between calls", self.name)
    }

    fn process(&mut self, tick: Tick) -> Result<Emitted> {
        #[cfg(feature = "gpu")]
        if self.textures.is_some() {
            return self.process_textures(tick);
        }
        let mut frames: Vec<RawFrame> = Vec::new();
        for stream in &tick.streams {
            for frame in &stream.frames {
                let bytes = self.bytes(&frame.data)?;
                self.framer.push(frame.pts, bytes, &mut frames)?;
            }
        }
        if tick.last {
            self.framer.finish(&mut frames);
        }
        let mut packets = if frames.is_empty() {
            Vec::new()
        } else {
            self.encoder.encode(&frames, false)?
        };
        if tick.last {
            packets.extend(self.encoder.encode(&[], true)?);
        }
        Ok(emitted(packets))
    }
}

/// How an encoder reading textures is handed `video`, whose frames arrive
/// in `arriving`: made its format on the device, in the colour its stream
/// says, each picture lasting what the stream's rate settles where it does.
#[cfg(feature = "gpu")]
fn textures_for(
    name: &str,
    arriving: &'static str,
    video: &runtime::VideoFormat,
    frame_rate: Option<(u64, u64)>,
    base: TimeBase,
) -> Result<Textures> {
    use ffrwd_wasm_runtime::gpu::frames::{Layout, Target};
    let layout = Layout::of(video.pix_fmt)
        .filter(|l| l.wire() == *l)
        .ok_or_else(|| {
            anyhow!(
                "{name} reads {}, which the device holds no frame in",
                video.pix_fmt
            )
        })?;
    Ok(Textures {
        arriving,
        target: Target::new(
            layout,
            video.width,
            video.height,
            av_nodes::yuv_of(video.color.as_ref()),
        ),
        rate_ticks: frame_ticks(frame_rate, base),
        held: None,
    })
}

#[cfg(feature = "gpu")]
impl EncoderNode {
    /// `process` for an encoder reading textures: each picture made the
    /// format it reads on the device and handed over as its planes.
    fn process_textures(&mut self, tick: Tick) -> Result<Emitted> {
        let textures = self.textures.as_mut().expect("an encoder reading textures");
        let mut pictures = Vec::new();
        for stream in &tick.streams {
            for frame in &stream.frames {
                textures
                    .push(frame.pts, &frame.data, &mut pictures)
                    .with_context(|| format!("{}: the picture at pts {}", self.name, frame.pts))?;
            }
        }
        if tick.last {
            textures.finish(&mut pictures);
        }
        let mut packets = if pictures.is_empty() {
            Vec::new()
        } else {
            self.encoder.encode_gpu(&pictures, false)?
        };
        if tick.last {
            packets.extend(self.encoder.encode_gpu(&[], true)?);
        }
        Ok(emitted(packets))
    }
}

/// An encoder's packets, on its one output.
fn emitted(packets: Vec<Packet>) -> Emitted {
    Emitted {
        items: packets
            .into_iter()
            .map(|packet| Emission {
                port: 0,
                payload: Payload::Packet(packet),
            })
            .collect(),
        rows: Vec::new(),
        finished: false,
    }
}

/// The decoder of a codec package: coded packets in, its frames out.
pub struct DecoderNode {
    name: String,
    shape: NodeShape,
    decoder: Decoder,
    format: Format,
    header: nut::Stream,
    /// How long a packet lasts, where the stream's rate settles it.
    duration: Option<i64>,
    written: u64,
    /// The pictures as its reader is handed them: the decoder's own, or
    /// made another format on the device.
    handed: StreamFormat,
    /// The format its pictures are made on the device, and the colour their
    /// YUV is read in, where its reader holds them there in another.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    convert: Option<(&'static str, av_nodes::Yuv)>,
    /// Where its reader holds the pictures on the GPU and the module writes
    /// textures: the colour their YUV is read in.
    #[cfg(feature = "gpu")]
    textures: Option<av_nodes::Yuv>,
}

impl DecoderNode {
    /// Opens the decoder of `path` on the coded stream `coded` carries under
    /// its NUT `header`, with `color` standing over what that says. `pix_fmt`
    /// is the format its reader holds the pictures in on the GPU, where they
    /// are made it after they are decoded: left there by a module that
    /// writes textures, else put there.
    pub fn open(
        path: &str,
        params: &str,
        coded: &nut::Stream,
        info: &StreamInfo,
        color: Option<runtime::ColorInfo>,
        pix_fmt: Option<&str>,
    ) -> Result<DecoderNode> {
        let mut decoder = Decoder::load(path).with_context(|| format!("opening module {path}"))?;
        let name = decoder.name().to_string();
        let mut stream = coded_input(coded)?;
        if let (runtime::CodedFormat::Video { color: said, .. }, Some(color)) =
            (&mut stream.format, color)
        {
            *said = Some(color);
        }
        let coded_color = match &stream.format {
            runtime::CodedFormat::Video { color, .. } => *color,
            _ => None,
        };
        let info = StreamInfo {
            codec: stream.codec.clone(),
            ..info.clone()
        };
        let decoded = decoder.init(&stream, &info, params)?;
        let (mut header, format) = raw_header(&name, &decoded, coded, coded_color.as_ref())?;
        #[cfg(feature = "gpu")]
        let textures = match format.media {
            Media::Video(video) if pix_fmt.is_some() && decoder.on_gpu() => {
                Some(av_nodes::yuv_of(video.color.as_ref()))
            }
            _ => None,
        };
        let mut convert = None;
        let handed = match format.media {
            Media::Video(mut video) => {
                if let Some(wanted) = pix_fmt.filter(|wanted| *wanted != video.pix_fmt) {
                    convert = Some(made_on_the_device(&name, wanted, &mut video, &mut header)?);
                }
                StreamFormat::Video(video)
            }
            Media::Audio(audio) => StreamFormat::Audio(audio),
        };
        let kind = match format.media {
            Media::Video(_) => PortKind::Video,
            Media::Audio(_) => PortKind::Audio,
        };
        Ok(DecoderNode {
            shape: av_nodes::shape(PortKind::Packets, kind, stream.time_base, 0.0),
            duration: frame_ticks(coded.frame_rate, stream.time_base),
            name,
            decoder,
            format,
            header,
            written: 0,
            handed,
            convert,
            #[cfg(feature = "gpu")]
            textures,
        })
    }

    /// A decoded picture as its reader is handed it: in host memory, or
    /// made the reader's format on the device.
    fn handed(&self, data: Vec<u8>) -> Result<FrameData> {
        #[cfg(feature = "gpu")]
        if let (Some((to, yuv)), Media::Video(video)) = (self.convert, self.format.media) {
            use ffrwd_wasm_runtime::gpu::frames::{GpuFrame, Layout, Target};
            let layout = Layout::of(to).expect("a wire format the device holds");
            let from = video.pix_fmt;
            let up = GpuFrame::upload(from, video.width, video.height, Arc::new(data))
                .with_context(|| format!("{}: putting a picture on the GPU", self.name))?;
            let made = up
                .converted(Target::new(layout, video.width, video.height, yuv))
                .with_context(|| format!("{}: making a picture {to} on the device", self.name))?;
            return Ok(FrameData::Gpu(made));
        }
        Ok(FrameData::Host(Arc::new(data)))
    }

    pub fn format(&self) -> StreamFormat {
        self.handed.clone()
    }

    pub fn header(&self) -> &nut::Stream {
        &self.header
    }

    pub fn time_base(&self) -> TimeBase {
        self.format.time_base
    }
}

impl Node for DecoderNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn shape(&self) -> &NodeShape {
        &self.shape
    }

    fn decode_delay(&self, _port: usize) -> Option<u32> {
        Some(0)
    }

    fn set_params(&mut self, _params: &str) -> Result<()> {
        bail!("{} takes no params between calls", self.name)
    }

    fn process(&mut self, tick: Tick) -> Result<Emitted> {
        let last = tick.last;
        let packets: Vec<Packet> = tick
            .streams
            .into_iter()
            .flat_map(|stream| stream.packets)
            .map(|packet| Packet {
                duration: packet.duration.or(self.duration),
                ..packet
            })
            .collect();
        #[cfg(feature = "gpu")]
        if let Some(yuv) = self.textures {
            return self.process_textures(packets, last, yuv);
        }
        let mut frames = if packets.is_empty() {
            Vec::new()
        } else {
            self.decoder.decode(&packets, false)?
        };
        if last {
            frames.extend(self.decoder.decode(&[], true)?);
        }
        let mut items = Vec::with_capacity(frames.len());
        for frame in frames {
            crate::check_frame(&self.format, &frame.data, self.written)
                .with_context(|| format!("{} wrote a frame at pts {}", self.name, frame.pts))?;
            self.written += 1;
            items.push(Emission {
                port: 0,
                payload: Payload::Frame(OutFrame {
                    pts: frame.pts,
                    duration: frame.duration,
                    data: self.handed(frame.data)?,
                    rows: Vec::new(),
                }),
            });
        }
        Ok(Emitted {
            items,
            rows: Vec::new(),
            finished: false,
        })
    }
}

#[cfg(feature = "gpu")]
impl DecoderNode {
    /// `process` for a decoder writing textures: each picture left on the
    /// device, made its reader's format there where that is another, its
    /// YUV read in `yuv`.
    fn process_textures(
        &mut self,
        packets: Vec<Packet>,
        last: bool,
        yuv: av_nodes::Yuv,
    ) -> Result<Emitted> {
        use ffrwd_wasm_runtime::gpu::frames::{Layout, Target};
        let mut pictures = if packets.is_empty() {
            Vec::new()
        } else {
            self.decoder.decode_gpu(&packets, false, yuv)?
        };
        if last {
            pictures.extend(self.decoder.decode_gpu(&[], true, yuv)?);
        }
        let mut items = Vec::with_capacity(pictures.len());
        for picture in pictures {
            let frame = match (self.convert, &self.handed) {
                (Some((to, yuv)), StreamFormat::Video(video)) => {
                    let layout = Layout::of(to).expect("a wire format the device holds");
                    picture
                        .frame
                        .converted(Target::new(layout, video.width, video.height, yuv))
                        .with_context(|| {
                            format!("{}: making a picture {to} on the device", self.name)
                        })?
                }
                _ => picture.frame,
            };
            self.written += 1;
            items.push(Emission {
                port: 0,
                payload: Payload::Frame(OutFrame {
                    pts: picture.pts,
                    duration: picture.duration,
                    data: FrameData::Gpu(frame),
                    rows: Vec::new(),
                }),
            });
        }
        Ok(Emitted {
            items,
            rows: Vec::new(),
            finished: false,
        })
    }
}

/// The format `wanted` a decoder's pictures, `video`, are made on the device,
/// with `header` saying it: the colour their YUV is read in is the
/// stream's, as libswscale reads it.
#[cfg(feature = "gpu")]
fn made_on_the_device(
    name: &str,
    wanted: &str,
    video: &mut runtime::VideoFormat,
    header: &mut nut::Stream,
) -> Result<(&'static str, av_nodes::Yuv)> {
    let to = ffrwd_wasm_runtime::av::WIRE_PIX_FMTS
        .iter()
        .copied()
        .find(|f| *f == wanted)
        .ok_or_else(|| anyhow!("{name}: {wanted} is not a format the wire carries"))?;
    let yuv = av_nodes::yuv_of(video.color.as_ref());
    let mut made = nut::Stream::video(to, video.width, video.height, header.time_base)
        .ok_or_else(|| anyhow!("{name}: the wire does not carry {to}"))?;
    made.frame_rate = header.frame_rate;
    if let (
        nut::Media::Video {
            sample_width,
            sample_height,
            ..
        },
        nut::Media::Video {
            sample_width: w,
            sample_height: h,
            ..
        },
    ) = (&mut made.media, header.media)
    {
        (*sample_width, *sample_height) = (w, h);
    }
    *header = made;
    video.pix_fmt = to;
    video.frame_len = crate::frame_len_for(to, video.width, video.height)?;
    Ok((to, yuv))
}

#[cfg(not(feature = "gpu"))]
fn made_on_the_device(
    name: &str,
    wanted: &str,
    video: &mut runtime::VideoFormat,
    _header: &mut nut::Stream,
) -> Result<(&'static str, av_nodes::Yuv)> {
    bail!(
        "{name} writes {}, and its reader takes {wanted}; this build holds no picture on a GPU \
         to make it there",
        video.pix_fmt
    )
}
