//! A codec package's module, hosted where ffmpeg's own encoder or decoder
//! would stand: one raw stream in and that stream coded out, or the mirror.
//!
//! ffmpeg knows no codec a package brings, so the coded side of either run
//! is a NUT stream under the codec's four-character tag, which ffmpeg
//! carries through `-c copy` without reading it. The raw side is the wire a
//! frame module reads and writes: rawvideo in a pixel format the wire
//! carries, or interleaved pcm.
//!
//! One instance, called in sequence: `-jobs` caps nothing here. A reader
//! thread demuxes the input so the calls are handed whatever has arrived in
//! one batch, and a pipe output is flushed after every batch.

use std::io;
use std::sync::mpsc;
use std::thread;

use anyhow::{anyhow, bail, Context, Result};
use ffrwd_wasm::nut;
use ffrwd_wasm_runtime::runtime::{
    self, CodedFormat, CodedStream, DecodedFormat, Decoder, Encoder, Format, Media, RawFrame,
    StreamInfo, TimeBase,
};

use super::{
    aspect_from, check_frame, color_from, colorspace_type_for, format_from_stream, frame_len_for,
    open_frame_output, open_input, output_is_pipe, stream_info, Args, FrameOutput, Input,
    OutputKind, EDGE_FORMAT,
};

/// `-codec`: which half of a codec package's module a run drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CodecHalf {
    Encode,
    Decode,
}

impl CodecHalf {
    /// `-codec`'s value, parsed.
    pub(crate) fn parse(raw: &str) -> Result<CodecHalf> {
        match raw {
            "encode" => Ok(CodecHalf::Encode),
            "decode" => Ok(CodecHalf::Decode),
            other => bail!("-codec {other}: only encode and decode are supported"),
        }
    }
}

/// `-frame_rate`'s value, `num/den` with both positive and within the
/// rational a module is handed: `30/1`, `30000/1001`.
pub(crate) fn parse_frame_rate(raw: &str) -> Result<(i32, i32)> {
    let parsed = raw.split_once('/').and_then(|(num, den)| {
        let num: i32 = num.trim().parse().ok()?;
        let den: i32 = den.trim().parse().ok()?;
        (num > 0 && den > 0).then_some((num, den))
    });
    parsed.ok_or_else(|| {
        anyhow!(
            "-frame_rate {raw}: a frame rate is <num>/<den>, both positive, as 30/1 or 30000/1001"
        )
    })
}

/// `-color_range`, `-color_primaries`, `-color_trc` and `-colorspace`: the
/// colorimetry of the stream a codec run reads, in ffmpeg's own names. The
/// NUT header ffmpeg writes carries none of it, so the caller says it here,
/// and each field it names stands over what the input's header declared.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ColorFlags {
    pub(crate) range: Option<&'static str>,
    pub(crate) primaries: Option<&'static str>,
    pub(crate) trc: Option<&'static str>,
    pub(crate) space: Option<&'static str>,
}

/// The spelling of "not said" in `ColorInfo`.
const UNKNOWN: &str = "unknown";

impl ColorFlags {
    /// Reads one of the four flags into its field, refusing it twice.
    pub(crate) fn set(&mut self, flag: &str, raw: &str) -> Result<()> {
        let slot = match flag {
            "-color_range" => &mut self.range,
            "-color_primaries" => &mut self.primaries,
            "-color_trc" => &mut self.trc,
            "-colorspace" => &mut self.space,
            other => bail!("{other} is not a colorimetry flag"),
        };
        if slot.is_some() {
            bail!("second {flag} specified");
        }
        *slot = Some(parse_color_name(flag, raw)?);
        Ok(())
    }

    /// True where no flag was given.
    pub(crate) fn is_empty(&self) -> bool {
        *self == ColorFlags::default()
    }

    /// `color` with every field a flag named put in its place. No flags
    /// leave it as it was; a flag over no colorimetry at all starts from
    /// every field unknown.
    pub(crate) fn over(&self, color: Option<runtime::ColorInfo>) -> Option<runtime::ColorInfo> {
        if self.is_empty() {
            return color;
        }
        let base = color.unwrap_or(runtime::ColorInfo {
            range: UNKNOWN,
            primaries: UNKNOWN,
            trc: UNKNOWN,
            space: UNKNOWN,
        });
        Some(runtime::ColorInfo {
            range: self.range.unwrap_or(base.range),
            primaries: self.primaries.unwrap_or(base.primaries),
            trc: self.trc.unwrap_or(base.trc),
            space: self.space.unwrap_or(base.space),
        })
    }
}

/// One colorimetry flag's value: an ffmpeg name, lowercase letters, digits,
/// `-` and `_`. The range is "tv" or "pc", the two ffprobe reports.
pub(crate) fn parse_color_name(flag: &str, raw: &str) -> Result<&'static str> {
    let name = raw.trim();
    let spelled = !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_');
    if !spelled {
        bail!("{flag} {raw}: a colorimetry value is one of ffmpeg's names, as bt709");
    }
    if flag == "-color_range" && !matches!(name, "tv" | "pc" | UNKNOWN) {
        bail!("{flag} {raw}: the range is tv or pc");
    }
    // Read once per run, so the few bytes live as long as the process.
    Ok(Box::leak(name.to_string().into_boxed_str()))
}

/// The most items one call is handed: whatever has arrived, up to this.
const BATCH: usize = 32;

/// How many demuxed items wait between the reader thread and the calls.
const QUEUE: usize = 64;

/// One item off the input: its frame header and its bytes.
type Item = (nut::Packet, Vec<u8>);

/// One read off the input: an item, or the end.
type Read = Result<Option<Item>>;

/// Runs the half of `module` the line asks for, or the only half it has.
pub(crate) fn run_codec(args: &Args, module: &str, params: &str) -> Result<()> {
    let encoder =
        runtime::exports_encoder(module).with_context(|| format!("opening module {module}"))?;
    let decoder =
        runtime::exports_decoder(module).with_context(|| format!("opening module {module}"))?;
    let half = match (args.codec, encoder, decoder) {
        (Some(CodecHalf::Encode), true, _) => CodecHalf::Encode,
        (Some(CodecHalf::Decode), _, true) => CodecHalf::Decode,
        (Some(CodecHalf::Encode), false, _) => {
            bail!("-codec encode: {module} exports a decoder and no encoder; it runs -codec decode")
        }
        (Some(CodecHalf::Decode), _, false) => {
            bail!(
                "-codec decode: {module} exports an encoder and no decoder; it runs -codec encode"
            )
        }
        (None, true, true) => bail!(
            "{module} exports both an encoder and a decoder; -codec encode or -codec decode \
             says which half runs"
        ),
        (None, true, false) => CodecHalf::Encode,
        (None, false, true) => CodecHalf::Decode,
        (None, false, false) => bail!("{module} exports neither an encoder nor a decoder"),
    };

    if half == CodecHalf::Decode && args.frame_rate.is_some() {
        bail!("-frame_rate is handed to an encoder, and this run decodes");
    }
    if args.annotations.input || args.annotations.output {
        bail!(
            "{module} is a codec: one stream in and one out, and no rows ride either, so \
             -annotations has nothing to carry"
        );
    }
    if args.pads.iter().any(Option::is_some) {
        bail!("-pad follows a packet sink's -i; {module} is a codec");
    }
    let [input] = args.inputs.as_slice() else {
        bail!(
            "{module} is a codec and reads one stream; this command gives it {} -i input(s)",
            args.inputs.len()
        );
    };
    let [output] = args.outputs.as_slice() else {
        bail!(
            "{module} is a codec and writes one stream; this command gives it {} outputs",
            args.outputs.len()
        );
    };
    if output.kind != OutputKind::Frames {
        bail!(
            "{}: {module} is a codec and writes one -f {EDGE_FORMAT} stream",
            output.spelling
        );
    }
    let reader = io::BufReader::with_capacity(1 << 20, open_input(input)?);
    let demuxer = nut::Demuxer::open(reader).context("reading the NUT input")?;
    match half {
        CodecHalf::Encode => encode(args, module, params, demuxer),
        CodecHalf::Decode => decode(args, module, params, demuxer),
    }
}

/// Demuxes `demuxer` on a thread of its own, so a call is handed whatever
/// has arrived rather than waiting on the wire item by item.
fn spawn_reader(mut demuxer: Input) -> mpsc::Receiver<Read> {
    let (tx, rx) = mpsc::sync_channel::<Read>(QUEUE);
    thread::spawn(move || loop {
        let mut data = Vec::new();
        let read = demuxer
            .read_packet(&mut data)
            .context("reading the NUT input")
            .map(|packet| packet.map(|p| (p, data)));
        let done = !matches!(read, Ok(Some(_)));
        if tx.send(read).is_err() || done {
            return;
        }
    });
    rx
}

/// The next batch: one item waited for, then whatever else has already
/// arrived, up to [`BATCH`]. The flag is whether the input has ended.
fn next_batch(rx: &mpsc::Receiver<Read>) -> Result<(Vec<Item>, bool)> {
    let mut batch = Vec::new();
    let first = rx
        .recv()
        .map_err(|_| anyhow!("the input reader stopped without saying why"))?;
    match first? {
        Some(item) => batch.push(item),
        None => return Ok((batch, true)),
    }
    while batch.len() < BATCH {
        match rx.try_recv() {
            Ok(read) => match read? {
                Some(item) => batch.push(item),
                None => return Ok((batch, true)),
            },
            Err(_) => break,
        }
    }
    Ok((batch, false))
}

/// How many ticks of `time_base` one frame at `frame_rate` lasts, when that
/// is a whole number; None when there is no rate or it does not divide.
fn frame_ticks(frame_rate: Option<(u64, u64)>, time_base: TimeBase) -> Option<i64> {
    let (fps_num, fps_den) = frame_rate?;
    let num = u128::from(time_base.den) * u128::from(fps_den);
    let den = u128::from(time_base.num) * u128::from(fps_num);
    (den != 0 && num % den == 0)
        .then(|| i64::try_from(num / den).ok())
        .flatten()
        .filter(|ticks| *ticks > 0)
}

/// `samples` at `rate` as ticks of `time_base`, rounded down.
fn sample_ticks(samples: u64, rate: u32, time_base: TimeBase) -> i64 {
    let num = u128::from(samples) * u128::from(time_base.den);
    let den = u128::from(time_base.num) * u128::from(rate);
    i64::try_from(num / den.max(1)).unwrap_or(i64::MAX)
}

/// `sample_ticks`, when it is exact; None where the time base cannot say
/// the length in whole ticks.
fn exact_sample_ticks(samples: u64, rate: u32, time_base: TimeBase) -> Option<i64> {
    let num = u128::from(samples) * u128::from(time_base.den);
    let den = u128::from(time_base.num) * u128::from(rate);
    (den != 0 && num % den == 0)
        .then(|| i64::try_from(num / den).ok())
        .flatten()
}

/// Cuts the raw input into the frames an encoder takes. Video is one frame
/// per picture; audio is one frame per packet when the encoder takes a run
/// of any length, and exactly `frame_samples` samples otherwise, the last
/// frame short.
struct Framer {
    format: Format,
    time_base: TimeBase,
    /// What a frame lasts, when the header's frame rate settles it.
    rate_ticks: Option<i64>,
    /// Video only, where no rate settles durations: the latest frame,
    /// held until the next one says how long it lasts.
    held: Option<(i64, Vec<u8>)>,
    frame_samples: usize,
    /// Audio being cut: the samples not yet handed on, the pts of the
    /// sample the run was anchored at, and how many samples since it have
    /// left.
    samples: Vec<u8>,
    anchor: i64,
    consumed: u64,
    index: u64,
}

impl Framer {
    fn new(format: Format, frame_rate: Option<(u64, u64)>, frame_samples: u32) -> Framer {
        Framer {
            format,
            time_base: format.time_base,
            rate_ticks: frame_ticks(frame_rate, format.time_base),
            held: None,
            frame_samples: frame_samples as usize,
            samples: Vec::new(),
            anchor: 0,
            consumed: 0,
            index: 0,
        }
    }

    /// One item off the wire, and the frames it completes.
    fn push(&mut self, pts: i64, data: Vec<u8>, out: &mut Vec<RawFrame>) -> Result<()> {
        check_frame(&self.format, &data, self.index)?;
        self.index += 1;
        match self.format.media {
            Media::Video(_) => {
                if let Some(ticks) = self.rate_ticks {
                    out.push(RawFrame {
                        pts,
                        duration: Some(ticks),
                        data,
                    });
                    return Ok(());
                }
                if let Some((held_pts, held)) = self.held.take() {
                    out.push(RawFrame {
                        pts: held_pts,
                        duration: Some(pts - held_pts).filter(|d| *d > 0),
                        data: held,
                    });
                }
                self.held = Some((pts, data));
            }
            Media::Audio(audio) => {
                let sample_len = audio.sample_len();
                if self.frame_samples == 0 {
                    let samples = (data.len() / sample_len) as u64;
                    out.push(RawFrame {
                        pts,
                        duration: exact_sample_ticks(samples, audio.sample_rate, self.time_base),
                        data,
                    });
                    return Ok(());
                }
                // A run is anchored at the pts of the packet that started
                // it and counted onward in samples; a packet reaching an
                // empty run starts a new one at its own pts.
                if self.samples.is_empty() {
                    self.anchor = pts;
                    self.consumed = 0;
                }
                self.samples.extend_from_slice(&data);
                let frame_len = self.frame_samples * sample_len;
                while self.samples.len() >= frame_len {
                    let rest = self.samples.split_off(frame_len);
                    let frame = std::mem::replace(&mut self.samples, rest);
                    out.push(self.cut(frame, audio.sample_rate, sample_len));
                }
            }
        }
        Ok(())
    }

    /// The audio frame `data` at the run's current position.
    fn cut(&mut self, data: Vec<u8>, rate: u32, sample_len: usize) -> RawFrame {
        let samples = (data.len() / sample_len) as u64;
        let pts = self.anchor + sample_ticks(self.consumed, rate, self.time_base);
        self.consumed += samples;
        RawFrame {
            pts,
            duration: exact_sample_ticks(samples, rate, self.time_base),
            data,
        }
    }

    /// Whatever is still held at the end of the input: the last picture,
    /// lasting what the rate says, or the short last run of samples.
    fn finish(&mut self, out: &mut Vec<RawFrame>) {
        if let Some((pts, data)) = self.held.take() {
            out.push(RawFrame {
                pts,
                duration: self.rate_ticks,
                data,
            });
        }
        if let Media::Audio(audio) = self.format.media {
            if !self.samples.is_empty() {
                let data = std::mem::take(&mut self.samples);
                out.push(self.cut(data, audio.sample_rate, audio.sample_len()));
            }
        }
    }
}

/// The coded output's NUT header: the encoder's tag, its time base,
/// extradata and reorder depth, and the geometry `init` answered. Where
/// the encoder answered no pixel aspect or colorimetry, the raw input's
/// pixel aspect and `input_color` are what describe the same pictures, so
/// they are carried through; so is the input's frame rate, which is the
/// only duration NUT has.
fn coded_header(
    name: &str,
    fourcc: &str,
    decode_delay: u32,
    coded: &CodedStream,
    raw: &nut::Stream,
    input_color: Option<&runtime::ColorInfo>,
) -> Result<nut::Stream> {
    let (kind, geometry) = match coded.format {
        CodedFormat::Video { width, height, .. } => ("video", (width, height)),
        CodedFormat::Audio {
            sample_rate,
            channels,
            ..
        } => ("audio", (sample_rate, channels)),
        CodedFormat::Data => {
            bail!("{name} answered a data stream; an encoder writes the kind it reads")
        }
    };
    let time_base = nut::TimeBase {
        num: coded.time_base.num,
        den: coded.time_base.den,
    };
    let mut stream = nut::Stream::coded_fourcc(
        kind,
        fourcc.as_bytes(),
        geometry,
        time_base,
        coded.extradata.clone(),
        u64::from(decode_delay),
    )
    .ok_or_else(|| {
        anyhow!(
            "{name} writes under the tag {fourcc}, which this wire carries as something other \
             than a {kind} codec of its own"
        )
    })?;
    if let (
        nut::Media::Video {
            sample_width,
            sample_height,
            colorspace_type,
            ..
        },
        CodedFormat::Video {
            sample_aspect_ratio,
            color,
            ..
        },
    ) = (&mut stream.media, &coded.format)
    {
        let (input_width, input_height) = match raw.media {
            nut::Media::Video {
                sample_width,
                sample_height,
                ..
            } => (sample_width, sample_height),
            _ => (1, 1),
        };
        (*sample_width, *sample_height) = sample_aspect_ratio
            .and_then(|(num, den)| (num > 0 && den > 0).then_some((num as u64, den as u64)))
            .unwrap_or((input_width, input_height));
        *colorspace_type = colorspace_type_for(color.as_ref().or(input_color));
    }
    stream.frame_rate = raw.frame_rate;
    Ok(stream)
}

/// Writes every packet of one call, in the decode order it answered them.
fn write_packets(out: &mut FrameOutput, packets: &[runtime::Packet], name: &str) -> Result<()> {
    for packet in packets {
        let framed = nut::Packet {
            pts: packet.pts,
            dts: packet.dts.or(Some(packet.pts)),
            keyframe: packet.keyframe,
        };
        out.write_coded(&framed, &packet.data)
            .with_context(|| format!("writing {name}'s packet at pts {}", packet.pts))?;
    }
    Ok(())
}

/// Raw frames in, the encoder's packets out.
fn encode(args: &Args, module: &str, params: &str, demuxer: Input) -> Result<()> {
    let mut encoder = Encoder::load(module).with_context(|| format!("opening module {module}"))?;
    let name = encoder.name().to_string();
    let raw = demuxer.stream().clone();
    if raw.pix_fmt().is_none() && raw.sample_fmt().is_none() {
        let carried = match raw.codec_name() {
            Some(codec) => format!("encoded {codec}"),
            None => format!("codec tag {}", raw.fourcc_name()),
        };
        bail!("{name} is an encoder and reads raw frames; the input carries {carried}");
    }
    let mut format = format_from_stream(&raw)?;
    let input_color = match &mut format.media {
        Media::Video(video) => {
            video.color = args.color.over(video.color);
            video.color
        }
        Media::Audio(_) if !args.color.is_empty() => {
            bail!("{name} is handed audio, and colorimetry describes pictures")
        }
        Media::Audio(_) => None,
    };
    let info = stream_info(args, &raw);
    let coded = encoder.init(&format, &info, args.frame_rate, params)?;
    let described = encoder.described().clone();
    let header = coded_header(
        &name,
        &described.fourcc,
        described.decode_delay,
        &coded,
        &raw,
        input_color.as_ref(),
    )?;
    let output = &args.outputs[0];
    let mut out = open_frame_output(&output.path, &header, false)
        .with_context(|| format!("opening output {}", output.spelling))?;
    let flush = output_is_pipe(&output.path);

    let mut framer = Framer::new(format, raw.frame_rate, described.frame_samples);
    let rx = spawn_reader(demuxer);
    let mut frames = Vec::new();
    loop {
        let (batch, ended) = next_batch(&rx)?;
        frames.clear();
        for (packet, data) in batch {
            framer.push(packet.pts, data, &mut frames)?;
        }
        if ended {
            framer.finish(&mut frames);
        }
        if !frames.is_empty() {
            let packets = encoder.encode(&frames, false)?;
            write_packets(&mut out, &packets, &name)?;
            if flush {
                out.flush()?;
            }
        }
        if ended {
            break;
        }
    }
    let packets = encoder.encode(&[], true)?;
    write_packets(&mut out, &packets, &name)?;
    out.finish()?;
    Ok(())
}

/// The coded stream a decoder is opened with, as the NUT header declared
/// it: its tag as text in `codec`, and no profile or level, which the wire
/// has no field for.
fn coded_input(stream: &nut::Stream) -> Result<CodedStream> {
    let format = match stream.media {
        nut::Media::Video {
            width,
            height,
            sample_width,
            sample_height,
            ..
        } => CodedFormat::Video {
            width,
            height,
            sample_aspect_ratio: aspect_from(sample_width, sample_height),
            color: color_from(&stream.media),
        },
        nut::Media::Audio {
            sample_rate,
            channels,
        } => CodedFormat::Audio {
            sample_rate,
            channels,
            channel_layout: None,
        },
        nut::Media::Other { .. } => bail!(
            "the input carries a {} stream; a decoder reads coded video or audio",
            stream.kind()
        ),
    };
    Ok(CodedStream {
        codec: stream.fourcc_name(),
        time_base: TimeBase {
            num: stream.time_base.num,
            den: stream.time_base.den,
        },
        format,
        extradata: stream.extradata.clone(),
        profile: None,
        level: None,
    })
}

/// The raw output's NUT header, from what the decoder's `init` answered:
/// the coded stream's time base, and its pixel aspect, frame rate and
/// (where the decoder declared none) `coded_color` carried through. Also
/// the host's format for the frames, which is what checks their sizes.
fn raw_header(
    name: &str,
    decoded: &DecodedFormat,
    coded: &nut::Stream,
    coded_color: Option<&runtime::ColorInfo>,
) -> Result<(nut::Stream, Format)> {
    let time_base = nut::TimeBase {
        num: coded.time_base.num,
        den: coded.time_base.den,
    };
    let host_time_base = TimeBase {
        num: time_base.num,
        den: time_base.den,
    };
    let mut stream = match decoded {
        DecodedFormat::Video {
            width,
            height,
            pix_fmt,
            color,
        } => {
            let mut stream =
                nut::Stream::video(pix_fmt, *width, *height, time_base).ok_or_else(|| {
                    anyhow!(
                        "{name} writes {pix_fmt}, and this wire carries {}",
                        nut::supported_pix_fmts().join(", ")
                    )
                })?;
            if let (
                nut::Media::Video {
                    sample_width,
                    sample_height,
                    colorspace_type,
                    ..
                },
                nut::Media::Video {
                    sample_width: coded_width,
                    sample_height: coded_height,
                    ..
                },
            ) = (&mut stream.media, coded.media)
            {
                if aspect_from(coded_width, coded_height).is_some() {
                    (*sample_width, *sample_height) = (coded_width, coded_height);
                }
                *colorspace_type = colorspace_type_for(color.as_ref().or(coded_color));
            }
            stream
        }
        DecodedFormat::Audio {
            sample_rate,
            channels,
            sample_fmt,
            ..
        } => {
            let mut stream =
                nut::Stream::audio(sample_fmt, *sample_rate, *channels).ok_or_else(|| {
                    anyhow!(
                        "{name} writes {sample_fmt} samples, and this wire carries {}",
                        nut::supported_sample_fmts().join(", ")
                    )
                })?;
            // Frames leave in the coded stream's time base, which need not
            // be the one tick per sample a fresh audio header counts in.
            stream.time_base = time_base;
            stream.max_pts_distance = time_base.den.div_ceil(time_base.num.max(1));
            stream
        }
    };
    stream.frame_rate = coded.frame_rate;
    let media = match decoded {
        DecodedFormat::Video {
            width,
            height,
            color,
            ..
        } => {
            let pix_fmt = stream.pix_fmt().expect("built from a carried pixel format");
            Media::Video(runtime::VideoFormat {
                width: *width,
                height: *height,
                pix_fmt,
                frame_len: frame_len_for(pix_fmt, *width, *height)
                    .with_context(|| format!("{name} writes {pix_fmt} frames"))?,
                color: *color,
            })
        }
        DecodedFormat::Audio {
            sample_rate,
            channels,
            ..
        } => Media::Audio(runtime::AudioFormat {
            sample_rate: *sample_rate,
            channels: *channels,
            sample_fmt: stream
                .sample_fmt()
                .expect("built from a carried sample format"),
            channel_layout: None,
        }),
    };
    Ok((
        stream,
        Format {
            media,
            time_base: host_time_base,
        },
    ))
}

/// Writes every frame of one call, each checked against the size the
/// decoder's own format says.
fn write_frames(
    out: &mut FrameOutput,
    frames: &[RawFrame],
    format: &Format,
    written: &mut u64,
    name: &str,
) -> Result<()> {
    for frame in frames {
        check_frame(format, &frame.data, *written)
            .with_context(|| format!("{name} wrote a frame at pts {}", frame.pts))?;
        out.write_frame(frame.pts, &frame.data)
            .with_context(|| format!("writing {name}'s frame at pts {}", frame.pts))?;
        *written += 1;
    }
    Ok(())
}

/// Coded packets in, the decoder's frames out.
fn decode(args: &Args, module: &str, params: &str, demuxer: Input) -> Result<()> {
    let mut decoder = Decoder::load(module).with_context(|| format!("opening module {module}"))?;
    let name = decoder.name().to_string();
    let stream = demuxer.stream().clone();
    if let Some(pix_fmt) = stream.pix_fmt() {
        bail!("{name} is a decoder and reads coded packets; the input carries raw {pix_fmt} video");
    }
    if let Some(sample_fmt) = stream.sample_fmt() {
        bail!(
            "{name} is a decoder and reads coded packets; the input carries raw {sample_fmt} audio"
        );
    }
    let mut coded = coded_input(&stream)?;
    let coded_color = match &mut coded.format {
        CodedFormat::Video { color, .. } => {
            *color = args.color.over(*color);
            *color
        }
        _ if !args.color.is_empty() => {
            bail!("{name} is handed audio, and colorimetry describes pictures")
        }
        _ => None,
    };
    let info = args.stream_info.clone().unwrap_or_else(|| StreamInfo {
        index: 0,
        kind: stream.kind().to_string(),
        codec: coded.codec.clone(),
        duration: None,
        tags: Vec::new(),
    });
    let decoded = decoder.init(&coded, &info, params)?;
    let (header, format) = raw_header(&name, &decoded, &stream, coded_color.as_ref())?;
    let output = &args.outputs[0];
    let mut out = open_frame_output(&output.path, &header, false)
        .with_context(|| format!("opening output {}", output.spelling))?;
    let flush = output_is_pipe(&output.path);

    // NUT has no per-packet duration; the frame rate is what settles one.
    let duration = frame_ticks(stream.frame_rate, coded.time_base);
    let rx = spawn_reader(demuxer);
    let mut written = 0u64;
    loop {
        let (batch, ended) = next_batch(&rx)?;
        if !batch.is_empty() {
            let packets: Vec<runtime::Packet> = batch
                .into_iter()
                .map(|(packet, data)| runtime::Packet {
                    pts: packet.pts,
                    dts: packet.dts,
                    duration,
                    keyframe: packet.keyframe,
                    data,
                })
                .collect();
            let frames = decoder.decode(&packets, false)?;
            write_frames(&mut out, &frames, &format, &mut written, &name)?;
            if flush {
                out.flush()?;
            }
        }
        if ended {
            break;
        }
    }
    let frames = decoder.decode(&[], true)?;
    write_frames(&mut out, &frames, &format, &mut written, &name)?;
    out.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{frame_ticks, parse_frame_rate, ColorFlags, Framer};
    use ffrwd_wasm_runtime::runtime::{
        AudioFormat, ColorInfo, Format, Media, RawFrame, TimeBase, VideoFormat,
    };

    #[test]
    fn a_colorimetry_flag_stands_over_the_header_field_by_field() {
        let header = ColorInfo {
            range: "tv",
            primaries: "unknown",
            trc: "unknown",
            space: "bt709",
        };
        assert_eq!(ColorFlags::default().over(Some(header)), Some(header));
        assert_eq!(ColorFlags::default().over(None), None);
        let mut flags = ColorFlags::default();
        flags.set("-color_primaries", "bt709").unwrap();
        flags.set("-color_range", "pc").unwrap();
        let expected = ColorInfo {
            range: "pc",
            primaries: "bt709",
            trc: "unknown",
            space: "bt709",
        };
        assert_eq!(flags.over(Some(header)), Some(expected));
        let alone = ColorInfo {
            space: "unknown",
            ..expected
        };
        assert_eq!(flags.over(None), Some(alone));
    }

    fn stereo_s16(time_base: TimeBase) -> Format {
        Format {
            media: Media::Audio(AudioFormat {
                sample_rate: 48000,
                channels: 2,
                sample_fmt: "s16",
                channel_layout: None,
            }),
            time_base,
        }
    }

    /// `samples` stereo s16 samples, each sample's bytes its own index so
    /// a cut in the wrong place shows.
    fn samples(from: u16, count: u16) -> Vec<u8> {
        (from..from + count)
            .flat_map(|i| {
                let [a, b] = i.to_le_bytes();
                [a, b, a, b]
            })
            .collect()
    }

    #[test]
    fn audio_is_cut_into_frames_of_the_encoders_size_the_last_one_short() {
        let tb = TimeBase { num: 1, den: 48000 };
        let mut framer = Framer::new(stereo_s16(tb), None, 1024);
        let mut out: Vec<RawFrame> = Vec::new();
        // 700 + 700 + 1000 samples: two whole frames and 352 left over.
        framer.push(0, samples(0, 700), &mut out).unwrap();
        assert!(out.is_empty(), "no whole frame yet");
        framer.push(700, samples(700, 700), &mut out).unwrap();
        framer.push(1400, samples(1400, 1000), &mut out).unwrap();
        framer.finish(&mut out);
        let cut: Vec<(i64, Option<i64>, usize)> = out
            .iter()
            .map(|f| (f.pts, f.duration, f.data.len() / 4))
            .collect();
        assert_eq!(
            cut,
            vec![
                (0, Some(1024), 1024),
                (1024, Some(1024), 1024),
                (2048, Some(352), 352)
            ]
        );
        let joined: Vec<u8> = out.iter().flat_map(|f| f.data.clone()).collect();
        assert_eq!(joined, samples(0, 2400), "every sample once, in order");
    }

    #[test]
    fn an_audio_run_counts_its_pts_in_the_streams_own_time_base() {
        // Milliseconds: 1024 samples at 48 kHz is 21.33 ms, so the pts are
        // rounded down and the duration, not a whole number of ticks, is
        // none rather than a guess.
        let tb = TimeBase { num: 1, den: 1000 };
        let mut framer = Framer::new(stereo_s16(tb), None, 1024);
        let mut out = Vec::new();
        framer.push(100, samples(0, 3072), &mut out).unwrap();
        let pts: Vec<i64> = out.iter().map(|f| f.pts).collect();
        assert_eq!(pts, vec![100, 121, 142]);
        assert!(out.iter().all(|f| f.duration.is_none()));
    }

    #[test]
    fn audio_of_any_length_goes_through_packet_for_packet() {
        let tb = TimeBase { num: 1, den: 48000 };
        let mut framer = Framer::new(stereo_s16(tb), None, 0);
        let mut out = Vec::new();
        framer.push(0, samples(0, 700), &mut out).unwrap();
        framer.push(700, samples(700, 3), &mut out).unwrap();
        framer.finish(&mut out);
        let cut: Vec<(i64, Option<i64>)> = out.iter().map(|f| (f.pts, f.duration)).collect();
        assert_eq!(cut, vec![(0, Some(700)), (700, Some(3))]);
    }

    fn video(time_base: TimeBase) -> Format {
        Format {
            media: Media::Video(VideoFormat {
                width: 2,
                height: 2,
                pix_fmt: "yuv420p",
                frame_len: 6,
                color: None,
            }),
            time_base,
        }
    }

    #[test]
    fn a_video_frame_lasts_until_the_next_one_where_no_rate_says() {
        let tb = TimeBase { num: 1, den: 90000 };
        let mut framer = Framer::new(video(tb), None, 0);
        let mut out = Vec::new();
        framer.push(0, vec![1; 6], &mut out).unwrap();
        assert!(out.is_empty(), "held until the next frame says how long");
        framer.push(3000, vec![2; 6], &mut out).unwrap();
        framer.push(9000, vec![3; 6], &mut out).unwrap();
        framer.finish(&mut out);
        let cut: Vec<(i64, Option<i64>, u8)> =
            out.iter().map(|f| (f.pts, f.duration, f.data[0])).collect();
        assert_eq!(
            cut,
            vec![(0, Some(3000), 1), (3000, Some(6000), 2), (9000, None, 3)]
        );
    }

    #[test]
    fn a_video_frame_leaves_at_once_where_the_rate_settles_its_length() {
        let tb = TimeBase { num: 1, den: 90000 };
        let mut framer = Framer::new(video(tb), Some((30, 1)), 0);
        let mut out = Vec::new();
        framer.push(0, vec![1; 6], &mut out).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].duration, Some(3000));
        assert_eq!(
            out[0].data,
            vec![1; 6],
            "the payload is handed through unchanged"
        );
    }

    #[test]
    fn frame_ticks_is_a_whole_number_or_none() {
        let tb = TimeBase { num: 1, den: 90000 };
        assert_eq!(frame_ticks(Some((30000, 1001)), tb), Some(3003));
        assert_eq!(frame_ticks(Some((7, 1)), tb), None);
        assert_eq!(frame_ticks(None, tb), None);
    }

    #[test]
    fn a_frame_rate_is_num_over_den_both_positive() {
        assert_eq!(parse_frame_rate("30/1").unwrap(), (30, 1));
        assert_eq!(parse_frame_rate("30000/1001").unwrap(), (30000, 1001));
        for bad in [
            "30",
            "30/0",
            "0/1",
            "-30/1",
            "a/b",
            "30/1/1",
            "",
            "4294967296/1",
        ] {
            let e = parse_frame_rate(bad).unwrap_err().to_string();
            assert!(e.contains("-frame_rate"), "{bad}: {e}");
        }
    }
}
