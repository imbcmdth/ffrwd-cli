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
//!
//! A `decode` whose pictures are read by an input held on the GPU, or by an
//! encoder that reads D3D12 pictures, decodes on the process's own device
//! where it can (`d3d12va` on Windows, `videotoolbox` on macOS), and hands
//! its pictures on as frames on that device, so nothing is copied to host
//! memory and back. Asked for rgba, it still leaves the decoder's NV12
//! there, and a reader of the frames is handed rgba converted on the device
//! once per picture.
//!
//! An `encode` whose encoder reads D3D12 pictures (`h264_d3d12va`,
//! `hevc_d3d12va`, `av1_d3d12va`) and whose frames arrive on the process's
//! device takes them there: each is copied on the device into a picture of
//! the encoder's pool, a module's rgba made NV12 first, and no byte crosses
//! to the host. Such an encoder reads nothing else, so a frame in host
//! memory is uploaded into its pool. Every other encoder reads a frame on
//! the GPU back once.
//!
//! `filter` runs a libavfilter graph on a D3D12 device -
//! `[d]filter=graph=scale_d3d12=w=1280\\:h=720[f]` - fed the way such an
//! encoder is: each frame on the process's device is copied there into a
//! picture of the graph's source, and one in host memory is uploaded. Its
//! pictures stay on the device as frames there, or, where the graph runs on
//! a device of its own, are read back.

use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use ffrwd_wasm::nut;
use ffrwd_wasm_runtime::av::{self, Decoder, Encoder, Hw, Received, WIRE_PIX_FMTS};
use ffrwd_wasm_runtime::node::{
    Accepts, Clock, Emission, Emitted, FrameData, FrameMemory, InputPort, Node, NodeShape,
    OutFrame, OutputPort, Pairing, Payload, PortKind, Rational, RowsUse, StreamFormat, Tick,
};
use ffrwd_wasm_runtime::runtime::{
    CodedFormat, CodedStream, ColorInfo, Packet, TimeBase, VideoFormat,
};

use crate::codec::frame_ticks;

/// The node that decodes.
pub const DECODE: &str = "decode";
/// The node that encodes.
pub const ENCODE: &str = "encode";
/// The node that runs a libavfilter graph.
pub const FILTER: &str = "filter";

const FORMAT: &str = "format";
const HW: &str = "hw";
const CODEC: &str = "codec";
const GRAPH: &str = "graph";

/// Whether `module` names `decode`, `encode` or `filter`.
pub fn is_av_node(module: &str) -> bool {
    module == DECODE || module == ENCODE || module == FILTER
}

/// The shape both have: one input by position, one output, ticking as
/// things arrive.
pub(crate) fn shape(input: PortKind, output: PortKind, base: TimeBase, latency: f64) -> NodeShape {
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

/// A colour as libav's shim takes it.
pub(crate) fn av_color(color: &ColorInfo) -> av::Color {
    av::Color {
        range: color.range.to_string(),
        primaries: color.primaries.to_string(),
        trc: color.trc.to_string(),
        space: color.space.to_string(),
    }
}

/// The colour a picture `decode` writes in `pix_fmt` is told, of a stream
/// in `color`: rgba is RGB, so full range with no matrix, in the stream's
/// primaries and transfer; YUV keeps the stream's.
fn written_color(pix_fmt: &str, color: Option<ColorInfo>) -> Option<ColorInfo> {
    if pix_fmt != "rgba" {
        return color;
    }
    let color = color.unwrap_or(ColorInfo {
        range: "unknown",
        primaries: "unknown",
        trc: "unknown",
        space: "unknown",
    });
    Some(ColorInfo {
        range: "pc",
        space: "gbr",
        ..color
    })
}

/// The decoder `decode` opens: on the process's own device when a reader
/// holds its pictures on the GPU and `hw` allows that device, else as `hw`
/// says. With it, how the pictures left on the device become frames.
/// `color` is the stream's, which a picture whose bitstream says none
/// carries.
fn open_decoder(
    codec: &str,
    extradata: &[u8],
    hw: Hw,
    want: &'static str,
    color: Option<&av::Color>,
    yuv: Option<Yuv>,
    on_gpu: bool,
) -> av::Result<(Decoder, Option<OnDevice>)> {
    if on_gpu {
        if let Some(opened) = open_on_shared_device(codec, extradata, &hw, want, color, yuv) {
            return Ok(opened);
        }
    }
    Ok((Decoder::open(codec, extradata, hw, want, color)?, None))
}

/// How a decoder's pictures on the process's device become frames.
#[cfg(feature = "gpu")]
pub(crate) type OnDevice = ffrwd_wasm_runtime::gpu::frames::DecodedPictures;
#[cfg(not(feature = "gpu"))]
pub(crate) type OnDevice = std::convert::Infallible;

/// The colour a picture's YUV is read in when it is made rgba on the
/// device.
#[cfg(feature = "gpu")]
pub(crate) type Yuv = ffrwd_wasm_runtime::gpu::frames::Yuv;
#[cfg(not(feature = "gpu"))]
pub(crate) type Yuv = ();

/// The colour a stream's YUV is read in as rgba, as libswscale reads it
/// (an unknown matrix is BT.601); None for a matrix the device's
/// conversion does not have.
#[cfg(feature = "gpu")]
pub(crate) fn read_as(color: Option<&ColorInfo>) -> Option<Yuv> {
    let (space, range) = color.map_or(("unknown", "unknown"), |c| (c.space, c.range));
    Yuv::read_as(space, range)
}

#[cfg(not(feature = "gpu"))]
pub(crate) fn read_as(_color: Option<&ColorInfo>) -> Option<Yuv> {
    None
}

/// The colour a stream's YUV is written and read in on the device: as
/// libswscale reads it, or BT.709 limited for a matrix the device's
/// conversion does not have.
#[cfg(feature = "gpu")]
pub(crate) fn yuv_of(color: Option<&ColorInfo>) -> Yuv {
    read_as(color).unwrap_or(Yuv::DEFAULT)
}

#[cfg(not(feature = "gpu"))]
pub(crate) fn yuv_of(_color: Option<&ColorInfo>) -> Yuv {}

/// A decoder on the process's own device, where `hw` allows the device
/// it is (`auto`, or that device by name: `d3d12va` on Windows,
/// `videotoolbox` on macOS), the wire's format is one its pictures are made
/// on the device (yuv420p, or rgba in a matrix the conversion has: `yuv`),
/// and the device opens. None, having said why on stderr where it is not
/// obvious, otherwise.
#[cfg(all(feature = "gpu", any(windows, target_os = "macos")))]
pub(crate) fn open_on_shared_device(
    codec: &str,
    extradata: &[u8],
    hw: &Hw,
    want: &'static str,
    color: Option<&av::Color>,
    yuv: Option<Yuv>,
) -> Option<(Decoder, Option<OnDevice>)> {
    use ffrwd_wasm_runtime::gpu;
    #[cfg(windows)]
    const DEVICE: &str = "d3d12va";
    #[cfg(windows)]
    const API: &str = "D3D12";
    #[cfg(target_os = "macos")]
    const DEVICE: &str = "videotoolbox";
    #[cfg(target_os = "macos")]
    const API: &str = "Metal";
    let allowed = match hw {
        Hw::Auto => true,
        Hw::Device(named) => named == DEVICE,
        Hw::None => false,
    };
    if !allowed {
        return None;
    }
    if !av::ON_DEVICE.contains(&want) {
        eprintln!(
            "[gpu] decode {codec}: {want} pictures are put on the GPU by their reader; a \
             decoder's own pictures stay there as {}",
            av::ON_DEVICE.join(" or ")
        );
        return None;
    }
    if want == "rgba" && yuv.is_none() {
        eprintln!(
            "[gpu] decode {codec}: the stream's matrix is not one rgba is made in on the \
             device (BT.709 or BT.601), so its pictures are converted in host memory and put \
             on the GPU by their reader"
        );
        return None;
    }
    #[cfg(windows)]
    let device = gpu::d3d12::av_device();
    #[cfg(target_os = "macos")]
    let device = gpu::metal::av_device();
    let Some(device) = device else {
        eprintln!(
            "[gpu] decode {codec}: the process's device is not a {API} one, so {DEVICE} \
             cannot share it"
        );
        return None;
    };
    match Decoder::open_shared(codec, extradata, device, want, color) {
        Ok(decoder) => Some((
            decoder,
            Some(gpu::frames::DecodedPictures::new(codec, DEVICE)),
        )),
        Err(e) => {
            eprintln!("[gpu] decode {codec}: not on the process's device: {e}");
            None
        }
    }
}

#[cfg(not(all(feature = "gpu", any(windows, target_os = "macos"))))]
pub(crate) fn open_on_shared_device(
    _codec: &str,
    _extradata: &[u8],
    _hw: &Hw,
    _want: &'static str,
    _color: Option<&av::Color>,
    _yuv: Option<Yuv>,
) -> Option<(Decoder, Option<OnDevice>)> {
    None
}

/// `decode`: one coded picture stream in, its pictures out in `format`.
pub struct Decode {
    name: String,
    shape: NodeShape,
    decoder: Decoder,
    /// Set when the decoder runs on the process's device and its pictures
    /// stay there.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    on_device: Option<OnDevice>,
    /// The pictures as the stream's header declares them, in the colour they
    /// are written in.
    format: VideoFormat,
    /// The colour the decoder's YUV is read in when a reader is handed
    /// rgba made on the device, until the decoder says what it found.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    yuv: Option<Yuv>,
    /// How long a picture lasts, where the frame rate says in whole ticks.
    duration: Option<i64>,
}

impl Decode {
    /// Opens the decoder for the coded stream `input` in `base`, carried
    /// under `fourcc` where its header names one. `delay` is the packets
    /// the header says the decoder holds back. `on_gpu` says an input that
    /// holds its frames on the GPU reads the pictures.
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        options: &[(String, String)],
        input: &StreamFormat,
        spelling: &str,
        fourcc: Option<&[u8]>,
        base: TimeBase,
        frame_rate: Option<(u64, u64)>,
        delay: u32,
        on_gpu: bool,
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
        let yuv = read_as(color.as_ref());
        let (decoder, on_device) = open_decoder(
            &codec,
            &coded.extradata,
            hw.unwrap_or_default(),
            want,
            color.as_ref().map(av_color).as_ref(),
            yuv,
            on_gpu,
        )
        .map_err(|e| anyhow!("{DECODE} {spelling}: {e}"))?;
        let format = VideoFormat {
            width,
            height,
            pix_fmt: want,
            frame_len: crate::frame_len_for(want, width, height)?,
            color: written_color(want, color),
        };
        // The pictures the decoder holds back, as seconds of the stream: a
        // lockstep reader is told to wait that long past a tick.
        let latency = match frame_rate {
            Some((num, den)) if num > 0 => f64::from(delay) * den as f64 / num as f64,
            _ => 0.0,
        };
        let mut shape = shape(PortKind::Packets, PortKind::Video, base, latency);
        if on_device.is_some() {
            shape.outputs[0].memory = FrameMemory::Gpu;
        }
        Ok(Decode {
            name: DECODE.to_string(),
            shape,
            decoder,
            on_device,
            format,
            yuv,
            duration: frame_ticks(frame_rate, base),
        })
    }

    /// The pictures this node writes, as the header declared them.
    pub fn format(&self) -> VideoFormat {
        self.format
    }

    /// Every picture the decoder has ready, onto `items`. A picture before
    /// zero is one the source's edit list decodes and never shows, as
    /// ffmpeg's own decode drops it.
    fn drain(&mut self, items: &mut Vec<Emission>) -> Result<()> {
        while let Some(received) = self
            .decoder
            .receive_any()
            .map_err(|e| anyhow!("{}: {e}", self.name))?
        {
            let (pts, duration, size) = match &received {
                Received::Host(f) => (f.pts, f.duration, (f.width, f.height)),
                Received::Device(f) => (f.pts, f.duration, (f.width, f.height)),
            };
            if pts < 0 {
                continue;
            }
            if size != (self.format.width, self.format.height) {
                bail!(
                    "{}: the stream's header says {}x{}, and the decoder made a {}x{} picture at \
                     pts {pts}; a stream that changes size mid-way is not carried",
                    self.name,
                    self.format.width,
                    self.format.height,
                    size.0,
                    size.1,
                );
            }
            let data = match received {
                Received::Host(frame) => Arc::new(frame.data).into(),
                Received::Device(picture) => self.on_device(picture)?,
            };
            items.push(Emission {
                port: 0,
                payload: Payload::Frame(OutFrame {
                    pts,
                    duration: duration.or(self.duration),
                    data,
                    rows: Vec::new(),
                }),
            });
        }
        Ok(())
    }

    /// A picture the decoder left on the process's device, as a frame there.
    #[cfg(feature = "gpu")]
    fn on_device(&mut self, picture: av::DeviceFrame) -> Result<FrameData> {
        let pictures = self.on_device.as_mut().ok_or_else(|| {
            anyhow!(
                "{}: a picture came back on a device the decoder was not opened on",
                self.name
            )
        })?;
        // The colour the decoder found, the bitstream's where it says one,
        // as the software decoder converts in.
        let yuv = self
            .decoder
            .format()
            .and_then(|found| Yuv::read_as(&found.color.space, &found.color.range))
            .or(self.yuv)
            .unwrap_or(Yuv::DEFAULT);
        let frame = pictures
            .hold(picture, self.format.pix_fmt, yuv)
            .map_err(|e| anyhow!("{}: {e:#}", self.name))?;
        Ok(FrameData::Gpu(frame))
    }

    #[cfg(not(feature = "gpu"))]
    fn on_device(&mut self, _picture: av::DeviceFrame) -> Result<FrameData> {
        bail!(
            "{}: a picture came back on the process's device, and this build holds none",
            self.name
        )
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
    /// what the decoder made of the stream after, each as its pictures are
    /// written.
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
            color: written_color(self.format.pix_fmt, Some(color)),
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
    /// Copies frames on the process's device into the encoder's pictures,
    /// for an encoder opened on that device. Dropped before the encoder,
    /// waiting for the copies still running into its pictures.
    #[cfg_attr(not(all(feature = "gpu", windows)), allow(dead_code))]
    feed: Option<Feed>,
    encoder: Encoder,
    coded: CodedStream,
    duration: Option<i64>,
    /// The colour an rgba frame is made NV12 in on the device: the stream's
    /// matrix and range, as the header declares them.
    #[cfg_attr(not(all(feature = "gpu", windows)), allow(dead_code))]
    yuv: ColorInfo,
    /// How frames last reached the encoder, said on stderr when it changes.
    said: Option<String>,
    /// How frames on the device were last copied into the encoder's
    /// pictures, said on stderr when it changes.
    #[cfg_attr(not(all(feature = "gpu", windows)), allow(dead_code))]
    copies: Option<String>,
}

#[cfg(all(feature = "gpu", windows))]
type Feed = ffrwd_wasm_runtime::gpu::d3d12::EncoderFeed;
#[cfg(not(all(feature = "gpu", windows)))]
type Feed = std::convert::Infallible;

/// The process's device for an encoder that reads D3D12 pictures, and the
/// feed that copies frames on it into the encoder's; None for both where
/// the process has no D3D12 device.
#[cfg(all(feature = "gpu", windows))]
fn encoder_device() -> Option<(av::Shared, Feed)> {
    Some((
        ffrwd_wasm_runtime::gpu::d3d12::av_device()?,
        Feed::default(),
    ))
}

#[cfg(not(all(feature = "gpu", windows)))]
fn encoder_device() -> Option<(av::Shared, Feed)> {
    None
}

/// What frames reach an encoder on the device as.
const FROM_THE_DEVICE: &str = "from the device";
const UPLOADED: &str = "uploaded from host memory";
const READ_BACK: &str = "read back from the GPU and uploaded";

/// The options `encode` reads as the colour its pictures are made and
/// tagged in, by ffmpeg's names, rather than handing them to the encoder.
const COLOR_OPTIONS: [&str; 4] = ["color_range", "color_primaries", "color_trc", "colorspace"];

/// `color` with each field `told` names in its place: the colour the
/// compiler settled for the pictures an encoder makes, where it settled one.
fn told_color(color: Option<ColorInfo>, told: &[(&str, &str)]) -> Result<Option<ColorInfo>> {
    if told.is_empty() {
        return Ok(color);
    }
    let mut made = color.unwrap_or(ColorInfo {
        range: "unknown",
        primaries: "unknown",
        trc: "unknown",
        space: "unknown",
    });
    for &(key, value) in told {
        let name = crate::codec::parse_color_name(&format!("-{key}"), value)
            .with_context(|| format!("{ENCODE} is told the colour it makes"))?;
        match key {
            "color_range" => made.range = name,
            "color_primaries" => made.primaries = name,
            "color_trc" => made.trc = name,
            _ => made.space = name,
        }
    }
    Ok(Some(made))
}

/// The colour an encoder that reads NV12 writes an rgba input in: the
/// stream's matrix and range where the conversion has them (`bt709`, or
/// BT.601's, which the wire calls `bt470bg`), else BT.709 limited range.
/// With it, why the stream's own was not kept, where it was not.
fn nv12_color(color: Option<ColorInfo>) -> (ColorInfo, Option<String>) {
    let unset = |name: &str| matches!(name, "" | "unknown" | "unspecified");
    let Some(color) = color else {
        return (
            ColorInfo {
                range: "tv",
                primaries: "unknown",
                trc: "unknown",
                space: "bt709",
            },
            None,
        );
    };
    let (space, why) = match color.space {
        "bt709" => ("bt709", None),
        "bt470bg" | "smpte170m" => ("bt470bg", None),
        other if unset(other) => ("bt709", None),
        other => ("bt709", Some(format!("the stream's matrix {other}"))),
    };
    let (range, why) = match color.range {
        "pc" | "jpeg" => ("pc", why),
        "tv" | "mpeg" => ("tv", why),
        other if unset(other) => ("tv", why),
        other => (
            "tv",
            why.or_else(|| Some(format!("the stream's range {other}"))),
        ),
    };
    (
        ColorInfo {
            range,
            space,
            ..color
        },
        why,
    )
}

impl Encode {
    /// Opens the encoder `codec=` names for the raw pictures `input` in
    /// `base`, handing it every other option by ffmpeg's name for it.
    /// `aspect` is the input's pixel aspect, which the coded stream
    /// declares in turn. `on_gpu` says the frames arrive held on the GPU,
    /// which an encoder that reads D3D12 pictures then takes on the
    /// process's device.
    ///
    /// Says on stderr what it opened, and for an encoder that reads D3D12
    /// pictures how frames reach it:
    /// `[av] encode h264_d3d12va: 1920x1080 nv12, from the device`.
    pub fn open(
        options: &[(String, String)],
        input: &StreamFormat,
        spelling: &str,
        base: TimeBase,
        frame_rate: Option<(u64, u64)>,
        aspect: Option<(i32, i32)>,
        on_gpu: bool,
    ) -> Result<Encode> {
        let mut codec: Option<&str> = None;
        let mut handed: Vec<(String, String)> = Vec::with_capacity(options.len());
        let mut told: Vec<(&str, &str)> = Vec::new();
        for (key, value) in options {
            if key == CODEC {
                if codec.is_some() {
                    bail!("{ENCODE} is given the option '{CODEC}' twice");
                }
                codec = Some(value);
            } else if COLOR_OPTIONS.contains(&key.as_str()) {
                told.push((key, value));
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
        let video = &VideoFormat {
            color: told_color(video.color, &told)?,
            ..*video
        };
        // An encoder that reads NV12 has an rgba input converted to it, in
        // the colour the stream's header then declares.
        let reads_d3d12 = Encoder::reads_d3d12(codec);
        let (yuv, stream_color) = if reads_d3d12 && video.pix_fmt == "rgba" {
            let (yuv, why) = nv12_color(video.color);
            if let Some(why) = why {
                eprintln!(
                    "[av] encode {codec}: rgba is made NV12 in BT.709 or BT.601 only, so \
                     {why} gives way to {} {}",
                    yuv.space, yuv.range
                );
            }
            (yuv, Some(yuv))
        } else {
            (nv12_color(video.color).0, video.color)
        };
        let color = stream_color.map(|c| av::Color {
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
        let (encoder, feed) = if reads_d3d12 {
            // On the process's device when the frames are there, so they
            // are copied on it; on a device of the encoder's own otherwise.
            let (device, feed) = match on_gpu.then(encoder_device).flatten() {
                Some((device, feed)) => (Some(device), Some(feed)),
                None => (None, None),
            };
            let encoder = Encoder::open_on_device(
                codec,
                video.width,
                video.height,
                video.pix_fmt,
                time_base,
                rate,
                color.as_ref(),
                &handed,
                device,
            );
            (encoder, feed)
        } else {
            let encoder = Encoder::open(
                codec,
                video.width,
                video.height,
                video.pix_fmt,
                time_base,
                rate,
                color.as_ref(),
                &handed,
            );
            (encoder, None)
        };
        let encoder = encoder.map_err(|e| anyhow!("{ENCODE} {spelling}: {e}"))?;
        let said = encoder.on_device().then(|| {
            if feed.is_some() {
                FROM_THE_DEVICE
            } else {
                UPLOADED
            }
            .to_string()
        });
        eprintln!(
            "[av] encode {}: {}x{} {}{}",
            encoder.name(),
            video.width,
            video.height,
            encoder.pix_fmt(),
            said.as_deref()
                .map(|how| format!(", {how}"))
                .unwrap_or_default()
        );
        let coded = CodedStream {
            codec: encoder.codec().to_string(),
            time_base: base,
            format: CodedFormat::Video {
                width: video.width,
                height: video.height,
                sample_aspect_ratio: aspect,
                color: stream_color,
            },
            extradata: encoder.extradata().to_vec(),
            profile: None,
            level: None,
        };
        Ok(Encode {
            name: ENCODE.to_string(),
            shape: shape(PortKind::Video, PortKind::Packets, base, 0.0),
            feed,
            encoder,
            coded,
            duration: frame_ticks(frame_rate, base),
            yuv,
            said,
            copies: None,
        })
    }

    /// Says on stderr how frames reach an encoder on the device, when that
    /// changes.
    fn say(&mut self, how: &str) {
        if self.said.as_deref() != Some(how) {
            eprintln!("[av] encode {}: {how}", self.encoder.name());
            self.said = Some(how.to_string());
        }
    }

    /// Hands one frame to the encoder: copied on the device into one of
    /// its pictures where the encoder reads its pool and the frame is on
    /// the process's device, else as bytes, read back where the frame is
    /// on the GPU.
    fn send(&mut self, data: &FrameData, pts: i64, duration: Option<i64>) -> Result<()> {
        if self.encoder.on_device() {
            if self.send_on_device(data, pts, duration)? {
                return Ok(());
            }
            self.say(match data {
                FrameData::Host(_) => UPLOADED,
                FrameData::Gpu(_) => READ_BACK,
            });
        }
        // A frame on the GPU is read back here, once, for any other encoder:
        // NVENC in this libav takes CUDA and D3D11 frames and no D3D12 ones.
        let bytes = data.bytes().map_err(|e| anyhow!("{}: {e}", self.name))?;
        self.encoder
            .send(pts, duration, &bytes)
            .map_err(|e| anyhow!("{}: {e}", self.name))
    }

    /// The frame copied on the device into one of the encoder's pictures
    /// and sent; false, with nothing sent, where it is not a frame on the
    /// process's device the encoder can be fed from.
    #[cfg(all(feature = "gpu", windows))]
    fn send_on_device(
        &mut self,
        data: &FrameData,
        pts: i64,
        duration: Option<i64>,
    ) -> Result<bool> {
        use ffrwd_wasm_runtime::gpu::d3d12::Copied;
        use ffrwd_wasm_runtime::gpu::frames::{Layout, Matrix};
        let (FrameData::Gpu(frame), Some(feed)) = (data, self.feed.as_mut()) else {
            return Ok(false);
        };
        if !frame.converts_to(Layout::Nv12) {
            return Ok(false);
        }
        let yuv = Yuv {
            matrix: if self.yuv.space == "bt709" {
                Matrix::Bt709
            } else {
                Matrix::Bt601
            },
            full_range: self.yuv.range == "pc",
        };
        let name = &self.name;
        let mut picture = self
            .encoder
            .device_picture()
            .map_err(|e| anyhow!("{name}: {e}"))?;
        let copied = feed
            .feed(frame, yuv, &mut picture)
            .map_err(|e| anyhow!("{name}: copying a frame into the encoder's: {e:#}"))?;
        self.encoder
            .send_picture(picture, pts, duration)
            .map_err(|e| anyhow!("{name}: {e}"))?;
        let how = match copied {
            Copied::Whole => "from the device, each frame one copy",
            Copied::Planes => "from the device, each frame a copy per plane",
        };
        let how = if frame.pix_fmt() == "nv12" {
            how.to_string()
        } else {
            format!("{how}, made NV12 from {} first", frame.pix_fmt())
        };
        self.say(FROM_THE_DEVICE);
        if self.copies.as_deref() != Some(how.as_str()) {
            eprintln!("[gpu] encode {}: {how}", self.encoder.name());
            self.copies = Some(how);
        }
        Ok(true)
    }

    #[cfg(not(all(feature = "gpu", windows)))]
    fn send_on_device(
        &mut self,
        _data: &FrameData,
        _pts: i64,
        _duration: Option<i64>,
    ) -> Result<bool> {
        Ok(false)
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
                self.send(&frame.data, frame.pts, frame.duration.or(self.duration))?;
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

/// `filter`: one raw picture stream through a libavfilter graph on the
/// device, its pictures out.
pub struct AvFilter {
    name: String,
    shape: NodeShape,
    /// Copies frames on the process's device into the graph's pictures.
    /// Dropped before the graph, waiting for the copies still running.
    #[cfg_attr(not(all(feature = "gpu", windows)), allow(dead_code))]
    feed: Option<FilterFeed>,
    filter: av::Filter,
    /// How the graph's pictures on the process's device become frames.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    on_device: Option<OnDevice>,
    /// The pictures as they arrive.
    input: VideoFormat,
    /// The pictures as they leave, as the graph's sink says.
    output: VideoFormat,
    /// The colour the YUV of a frame is read in where rgba is made NV12 on
    /// the way in, or a picture is read as rgba on the way out.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    yuv: Option<Yuv>,
    /// How long a picture lasts, in the input's ticks and in the output's,
    /// where the frame rate says in whole ticks.
    in_duration: Option<i64>,
    duration: Option<i64>,
    /// How frames last reached the graph, said on stderr when it changes.
    said: Option<String>,
}

#[cfg(all(feature = "gpu", windows))]
type FilterFeed = ffrwd_wasm_runtime::gpu::d3d12::FilterFeed;
#[cfg(not(all(feature = "gpu", windows)))]
type FilterFeed = std::convert::Infallible;

/// The process's device for a filter graph, and the feed that copies frames
/// on it into the graph's; None for both where the process has no D3D12
/// device.
#[cfg(all(feature = "gpu", windows))]
fn filter_device() -> Option<(av::Shared, FilterFeed)> {
    Some((
        ffrwd_wasm_runtime::gpu::d3d12::av_device()?,
        FilterFeed::default(),
    ))
}

#[cfg(not(all(feature = "gpu", windows)))]
fn filter_device() -> Option<(av::Shared, FilterFeed)> {
    None
}

/// What frames reach a graph as.
const COPIED_ON_THE_DEVICE: &str = "frames copied on the device";

impl AvFilter {
    /// Opens the graph `graph=` names for the raw pictures `input` in
    /// `base`, its pictures leaving packed as `format=` (yuv420p unless
    /// said). `on_gpu` says the frames arrive held on the GPU or a reader
    /// holds them there, which puts the graph on the process's own device
    /// where it is a D3D12 one, and its pictures stay there.
    ///
    /// Says on stderr, once, where the graph runs:
    /// `[av] filter scale_d3d12: on the host's device`.
    pub fn open(
        options: &[(String, String)],
        input: &StreamFormat,
        spelling: &str,
        base: TimeBase,
        frame_rate: Option<(u64, u64)>,
        on_gpu: bool,
    ) -> Result<AvFilter> {
        let mut graph: Option<&str> = None;
        let mut want: Option<&'static str> = None;
        for (key, value) in options {
            match key.as_str() {
                GRAPH if graph.is_some() => bail!("{FILTER} is given the option '{GRAPH}' twice"),
                GRAPH => graph = Some(value),
                FORMAT if want.is_some() => bail!("{FILTER} is given the option '{FORMAT}' twice"),
                FORMAT => {
                    let Some(found) = WIRE_PIX_FMTS.iter().copied().find(|f| f == value) else {
                        bail!(
                            "{FILTER}={FORMAT}={value}: a filter writes one of {}",
                            WIRE_PIX_FMTS.join(", ")
                        );
                    };
                    want = Some(found);
                }
                other => bail!(
                    "{FILTER} has no option '{other}'; it takes {GRAPH}=<libavfilter graph> and \
                     {FORMAT}=<pix_fmt> ({})",
                    WIRE_PIX_FMTS.join(", ")
                ),
            }
        }
        let Some(graph) = graph.filter(|g| !g.trim().is_empty()) else {
            bail!(
                "{FILTER} names the libavfilter graph it runs with {GRAPH}=<graph> \
                 (graph=scale_d3d12=w=1280\\\\:h=720); this call names none"
            );
        };
        let want = want.unwrap_or("yuv420p");
        let StreamFormat::Video(video) = input else {
            bail!(
                "{FILTER} reads raw pictures, and {spelling} is {}",
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
        let (device, feed) = match on_gpu.then(filter_device).flatten() {
            Some((device, feed)) => (Some(device), Some(feed)),
            None => (None, None),
        };
        let filter = av::Filter::open(
            graph,
            video.width,
            video.height,
            time_base,
            rate,
            color.as_ref(),
            device,
        )
        .map_err(|e| anyhow!("{FILTER} {spelling}: {e}"))?;
        let names = filter.filters().join(", ");
        let on_device = if filter.on_shared_device() {
            Some(filter_pictures(&names)?)
        } else {
            None
        };
        eprintln!(
            "[av] filter {names}: {}",
            if on_device.is_some() {
                "on the host's device".to_string()
            } else {
                "on a d3d12va device of its own, pictures read back to host memory".to_string()
            }
        );
        let (width, height) = filter.output_size();
        let (tb_num, tb_den) = filter.time_base();
        let out_base = match (u64::try_from(tb_num), u64::try_from(tb_den)) {
            (Ok(num), Ok(den)) if num > 0 && den > 0 => TimeBase { num, den },
            _ => base,
        };
        let out_rate = filter
            .frame_rate()
            .and_then(|(num, den)| Some((u64::try_from(num).ok()?, u64::try_from(den).ok()?)))
            .or(frame_rate);
        let output = VideoFormat {
            width,
            height,
            pix_fmt: want,
            frame_len: crate::frame_len_for(want, width, height)?,
            color: video.color,
        };
        let mut shape = shape(PortKind::Video, PortKind::Video, out_base, 0.0);
        if on_device.is_some() {
            shape.outputs[0].memory = FrameMemory::Gpu;
        }
        Ok(AvFilter {
            name: FILTER.to_string(),
            shape,
            feed,
            filter,
            on_device,
            input: *video,
            output,
            yuv: read_as(video.color.as_ref()),
            in_duration: frame_ticks(frame_rate, base),
            duration: frame_ticks(out_rate, out_base),
            said: None,
        })
    }

    /// The pictures this node writes, as the graph's sink declares them.
    pub fn format(&self) -> VideoFormat {
        self.output
    }

    /// The time base and frame rate of the pictures this node writes.
    pub fn timing(&self, frame_rate: Option<(u64, u64)>) -> (TimeBase, Option<(u64, u64)>) {
        let base = self.shape.outputs[0]
            .time_base
            .and_then(|r| {
                Some(TimeBase {
                    num: u64::try_from(r.num).ok()?,
                    den: u64::try_from(r.den).ok()?,
                })
            })
            .unwrap_or(TimeBase { num: 1, den: 1 });
        let rate = self
            .filter
            .frame_rate()
            .and_then(|(num, den)| Some((u64::try_from(num).ok()?, u64::try_from(den).ok()?)))
            .or(frame_rate);
        (base, rate)
    }

    /// Says on stderr how frames reach the graph, when that changes.
    fn say(&mut self, how: &str) {
        if self.said.as_deref() != Some(how) {
            eprintln!("[av] filter {}: {how}", self.filter.filters().join(", "));
            self.said = Some(how.to_string());
        }
    }

    /// Hands one frame to the graph: copied on the device into one of its
    /// pictures where the frame is on the process's device and the graph
    /// runs there, else as bytes, read back where the frame is on the GPU
    /// and uploaded.
    fn send(&mut self, data: &FrameData, pts: i64, duration: Option<i64>) -> Result<()> {
        if self.send_on_device(data, pts, duration)? {
            self.say(COPIED_ON_THE_DEVICE);
            return Ok(());
        }
        self.say(match data {
            FrameData::Host(_) => UPLOADED,
            FrameData::Gpu(_) => READ_BACK,
        });
        let bytes = data.bytes().map_err(|e| anyhow!("{}: {e}", self.name))?;
        self.filter
            .send(self.input.pix_fmt, pts, duration, &bytes)
            .map_err(|e| anyhow!("{}: {e}", self.name))
    }

    /// The frame copied on the device into one of the graph's pictures and
    /// sent; false, with nothing sent, where it is not a frame on the
    /// process's device the graph can be fed from.
    #[cfg(all(feature = "gpu", windows))]
    fn send_on_device(
        &mut self,
        data: &FrameData,
        pts: i64,
        duration: Option<i64>,
    ) -> Result<bool> {
        use ffrwd_wasm_runtime::gpu::frames::Layout;
        let (FrameData::Gpu(frame), Some(feed)) = (data, self.feed.as_mut()) else {
            return Ok(false);
        };
        if !frame.converts_to(Layout::Nv12) {
            return Ok(false);
        }
        let name = &self.name;
        let mut picture = self
            .filter
            .device_picture()
            .map_err(|e| anyhow!("{name}: {e}"))?;
        feed.feed(frame, self.yuv.unwrap_or(Yuv::DEFAULT), &mut picture)
            .map_err(|e| anyhow!("{name}: copying a frame into the graph's: {e:#}"))?;
        self.filter
            .send_picture(picture, pts, duration)
            .map_err(|e| anyhow!("{name}: {e}"))?;
        Ok(true)
    }

    #[cfg(not(all(feature = "gpu", windows)))]
    fn send_on_device(
        &mut self,
        _data: &FrameData,
        _pts: i64,
        _duration: Option<i64>,
    ) -> Result<bool> {
        Ok(false)
    }

    /// Every picture the graph has ready, onto `items`.
    fn drain(&mut self, items: &mut Vec<Emission>) -> Result<()> {
        while let Some(received) = self
            .filter
            .receive_any(self.output.pix_fmt)
            .map_err(|e| anyhow!("{}: {e}", self.name))?
        {
            let (pts, duration) = match &received {
                Received::Host(f) => (f.pts, f.duration),
                Received::Device(f) => (f.pts, f.duration),
            };
            let data = match received {
                Received::Host(frame) => Arc::new(frame.data).into(),
                Received::Device(picture) => self.on_device(picture)?,
            };
            items.push(Emission {
                port: 0,
                payload: Payload::Frame(OutFrame {
                    pts,
                    duration: duration.or(self.duration),
                    data,
                    rows: Vec::new(),
                }),
            });
        }
        Ok(())
    }

    /// A picture the graph left on the process's device, as a frame there.
    #[cfg(feature = "gpu")]
    fn on_device(&mut self, picture: av::DeviceFrame) -> Result<FrameData> {
        #[cfg(windows)]
        if let Some(feed) = self.feed.as_mut() {
            feed.wrote(&picture)
                .map_err(|e| anyhow!("{}: {e:#}", self.name))?;
        }
        let pictures = self.on_device.as_mut().ok_or_else(|| {
            anyhow!(
                "{}: a picture came back on a device the graph was not opened on",
                self.name
            )
        })?;
        let yuv = self.yuv.unwrap_or(Yuv::DEFAULT);
        let frame = pictures
            .hold(picture, self.output.pix_fmt, yuv)
            .map_err(|e| anyhow!("{}: {e:#}", self.name))?;
        Ok(FrameData::Gpu(frame))
    }

    #[cfg(not(feature = "gpu"))]
    fn on_device(&mut self, _picture: av::DeviceFrame) -> Result<FrameData> {
        bail!(
            "{}: a picture came back on the process's device, and this build holds none",
            self.name
        )
    }
}

/// How a graph's pictures on the process's device become frames.
#[cfg(feature = "gpu")]
fn filter_pictures(names: &str) -> Result<OnDevice> {
    Ok(ffrwd_wasm_runtime::gpu::frames::DecodedPictures::for_filter(names, "d3d12va"))
}

#[cfg(not(feature = "gpu"))]
fn filter_pictures(_names: &str) -> Result<OnDevice> {
    bail!("this build holds no frame on the GPU")
}

impl Drop for AvFilter {
    /// Waits for the device to finish reading the pictures the graph wrote
    /// that the host still keeps, so none is written again under a read.
    fn drop(&mut self) {
        #[cfg(all(feature = "gpu", windows))]
        if self.on_device.is_some() {
            ffrwd_wasm_runtime::gpu::d3d12::release_kept();
        }
    }
}

impl Node for AvFilter {
    fn name(&self) -> &str {
        &self.name
    }

    fn shape(&self) -> &NodeShape {
        &self.shape
    }

    fn settled_format(&self, _port: usize) -> Option<StreamFormat> {
        Some(StreamFormat::Video(self.output))
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
            for frame in stream.frames {
                self.send(&frame.data, frame.pts, frame.duration.or(self.in_duration))?;
                self.drain(&mut items)?;
            }
        }
        if tick.last {
            self.filter
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

    #[test]
    fn a_decoded_picture_is_told_the_colour_it_is_written_in() {
        let tv = ColorInfo {
            range: "tv",
            primaries: "bt709",
            trc: "bt709",
            space: "bt709",
        };
        let rgb = |primaries, trc| ColorInfo {
            range: "pc",
            primaries,
            trc,
            space: "gbr",
        };
        let cases = [
            ("yuv420p", Some(tv), Some(tv)),
            ("yuv420p", None, None),
            ("rgba", Some(tv), Some(rgb("bt709", "bt709"))),
            ("rgba", None, Some(rgb("unknown", "unknown"))),
        ];
        for (pix_fmt, stream, told) in cases {
            assert_eq!(
                written_color(pix_fmt, stream),
                told,
                "{pix_fmt} of {stream:?}"
            );
        }
    }

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
