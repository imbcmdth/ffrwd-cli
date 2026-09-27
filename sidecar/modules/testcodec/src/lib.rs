//! The smallest codec a package can bring: every frame one keyframe packet,
//! its bytes run-length coded, and a decoder that inverts it exactly.
//!
//! It is what the codec worlds are tested against. The host knows no codec
//! by this name, so what crosses the wire is its tag, FTST, and a round trip
//! through the encoder and the decoder must hand back the frames that went
//! in, byte for byte and at the same timestamps.
//!
//! A packet is a header and a body. The header is the magic `FT`, the width
//! and height as little-endian u16, and one byte naming the pixel format
//! (0 yuv420p, 1 gray, 2 yuv444p, 3 yuv422p). The body is runs: a count from
//! 1 to 255, then the byte repeated that many times.
//!
//! The extradata is the tag, a version byte and the pixel format's byte, so
//! a decoder opened on the stream writes the format that was coded. A header
//! without the format's byte decodes as yuv420p.

wit_bindgen::generate!({
    path: "../../wit",
    world: "codec-module",
});

use std::cell::RefCell;

use crate::ffrwd::av::types::{
    CodedFormat, CodedStream, CodedVideo, Format, Meta, Packet, Rational, RawFrame, StreamInfo,
    VideoFormat,
};
use exports::ffrwd::av::decoder::{self, DecoderMeta};
use exports::ffrwd::av::encoder::{self, EncoderMeta};

const NAME: &str = "testcodec";
const FOURCC: &str = "FTST";
/// The codec's out-of-band header: its tag and a version byte, which the
/// pixel format's byte follows.
const EXTRADATA: &[u8] = b"FTST\x01";
const MAGIC: &[u8; 2] = b"FT";
const HEADER_LEN: usize = 7;
const PIXEL_FORMATS: [&str; 4] = ["yuv420p", "gray", "yuv444p", "yuv422p"];

const ENCODER_PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"level":{"type":"number","minimum":0,"maximum":9}},"additionalProperties":false}"#;
const DECODER_PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"pix_fmt":{"type":"string","enum":["yuv420p","gray","yuv444p","yuv422p"]}},"additionalProperties":false}"#;

/// The geometry an instance was opened for.
#[derive(Clone, Copy)]
struct Opened {
    width: u32,
    height: u32,
    pix: u8,
}

thread_local! {
    static ENCODING: RefCell<Option<Opened>> = const { RefCell::new(None) };
    static DECODING: RefCell<Option<Opened>> = const { RefCell::new(None) };
}

/// This codec's byte for a pixel format, or None for one it does not code.
fn pix_code(pix_fmt: &str) -> Option<u8> {
    PIXEL_FORMATS
        .iter()
        .position(|f| *f == pix_fmt)
        .map(|i| i as u8)
}

/// Bytes in one frame of `pix` at this size.
fn frame_len(opened: Opened) -> usize {
    let pixels = opened.width as usize * opened.height as usize;
    match opened.pix {
        0 => pixels * 3 / 2,
        2 => pixels * 3,
        3 => pixels * 2,
        _ => pixels,
    }
}

fn meta(params_schema: &str) -> Meta {
    Meta {
        name: NAME.to_string(),
        version: "0.1.0".to_string(),
        params_schema: params_schema.to_string(),
        rows_schema: String::new(),
        pixel_formats: PIXEL_FORMATS.iter().map(|f| f.to_string()).collect(),
        sample_formats: vec![],
        sample_rates: vec![],
        channel_counts: vec![],
        rows_language: vec![],
    }
}

/// The params as a JSON object; empty is none.
fn params_object(params: &str) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    if params.trim().is_empty() {
        return Ok(serde_json::Map::new());
    }
    match serde_json::from_str(params) {
        Ok(serde_json::Value::Object(map)) => Ok(map),
        Ok(other) => Err(format!("{NAME} params are a JSON object, got: {other}")),
        Err(e) => Err(format!("{NAME} params are not JSON: {e}")),
    }
}

/// The encoder's params: `level`, a number from 0 to 9, taken and ignored.
fn check_encoder_params(params: &str) -> Result<(), String> {
    for (key, value) in params_object(params)? {
        if key != "level" {
            return Err(format!("{NAME} encoder takes level alone, got: {key}"));
        }
        match value.as_f64() {
            Some(level) if (0.0..=9.0).contains(&level) => {}
            _ => {
                return Err(format!(
                    "{NAME} level is a number from 0 to 9, got: {value}"
                ))
            }
        }
    }
    Ok(())
}

/// Runs of one byte: a count from 1 to 255, then the byte.
fn compress(data: &[u8], out: &mut Vec<u8>) {
    let mut i = 0;
    while i < data.len() {
        let byte = data[i];
        let mut run = 1;
        while run < 255 && i + run < data.len() && data[i + run] == byte {
            run += 1;
        }
        out.push(run as u8);
        out.push(byte);
        i += run;
    }
}

/// The inverse of `compress`, refusing a body that does not come out at
/// exactly `len` bytes.
fn expand(body: &[u8], len: usize) -> Result<Vec<u8>, String> {
    if !body.len().is_multiple_of(2) {
        return Err(format!(
            "{NAME}: a packet body is pairs, and this one is {} bytes",
            body.len()
        ));
    }
    let mut out = Vec::with_capacity(len);
    for pair in body.as_chunks::<2>().0 {
        if pair[0] == 0 {
            return Err(format!("{NAME}: a run of 0 bytes"));
        }
        out.extend(std::iter::repeat_n(pair[1], pair[0] as usize));
        if out.len() > len {
            return Err(format!(
                "{NAME}: a packet expands past the {len} bytes of a frame"
            ));
        }
    }
    if out.len() != len {
        return Err(format!(
            "{NAME}: a packet expands to {} bytes, not the {len} of a frame",
            out.len()
        ));
    }
    Ok(out)
}

struct TestCodec;

impl encoder::Guest for TestCodec {
    fn describe() -> EncoderMeta {
        EncoderMeta {
            meta: meta(ENCODER_PARAMS_SCHEMA),
            codec: NAME.to_string(),
            fourcc: FOURCC.to_string(),
            delay: 0,
            decode_delay: 0,
            frame_samples: 0,
        }
    }

    fn init(
        format: Format,
        info: StreamInfo,
        frame_rate: Option<Rational>,
        params: String,
    ) -> Result<CodedStream, String> {
        check_encoder_params(&params)?;
        // Taken and ignored, as a codec coding every frame alone may: a
        // host hands none for audio, and a positive rate otherwise.
        if let Some(rate) = frame_rate {
            if rate.num <= 0 || rate.den <= 0 {
                return Err(format!(
                    "{NAME}: frame rate {}/{} is not positive",
                    rate.num, rate.den
                ));
            }
        }
        let Format::Video(VideoFormat {
            width,
            height,
            pix_fmt,
            color,
        }) = format
        else {
            return Err(format!("{NAME} codes video alone"));
        };
        let pix = pix_code(&pix_fmt)
            .ok_or_else(|| format!("{NAME} codes {}, not {pix_fmt}", PIXEL_FORMATS.join(", ")))?;
        if width > u32::from(u16::MAX) || height > u32::from(u16::MAX) {
            return Err(format!(
                "{NAME} codes at most 65535x65535, not {width}x{height}"
            ));
        }
        ENCODING.with(|e| *e.borrow_mut() = Some(Opened { width, height, pix }));
        Ok(CodedStream {
            codec: NAME.to_string(),
            time_base: info.time_base,
            format: CodedFormat::Video(CodedVideo {
                width,
                height,
                sample_aspect_ratio: None,
                color,
            }),
            extradata: [EXTRADATA, &[pix]].concat(),
            profile: None,
            level: None,
        })
    }

    fn encode(frames: Vec<RawFrame>, _last: bool) -> Result<Vec<Packet>, String> {
        let opened = ENCODING
            .with(|e| *e.borrow())
            .ok_or_else(|| format!("{NAME}: encode before init"))?;
        let len = frame_len(opened);
        let mut packets = Vec::with_capacity(frames.len());
        for frame in frames {
            if frame.data.len() != len {
                return Err(format!(
                    "{NAME}: a frame of {} bytes, not the {len} this geometry is",
                    frame.data.len()
                ));
            }
            let mut data = Vec::with_capacity(HEADER_LEN + len / 4);
            data.extend_from_slice(MAGIC);
            data.extend_from_slice(&(opened.width as u16).to_le_bytes());
            data.extend_from_slice(&(opened.height as u16).to_le_bytes());
            data.push(opened.pix);
            compress(&frame.data, &mut data);
            packets.push(Packet {
                pts: frame.pts,
                dts: Some(frame.pts),
                duration: frame.duration,
                keyframe: true,
                data,
            });
        }
        Ok(packets)
    }
}

impl decoder::Guest for TestCodec {
    fn describe() -> DecoderMeta {
        DecoderMeta {
            meta: meta(DECODER_PARAMS_SCHEMA),
            fourccs: vec![FOURCC.to_string()],
            delay: 0,
        }
    }

    fn init(coded: CodedStream, _info: StreamInfo, params: String) -> Result<Format, String> {
        if coded.codec != FOURCC {
            return Err(format!("{NAME} reads {FOURCC}, not {}", coded.codec));
        }
        let coded_pix = match coded.extradata.strip_prefix(EXTRADATA) {
            Some([]) => None,
            Some([pix]) if usize::from(*pix) < PIXEL_FORMATS.len() => Some(*pix),
            _ => {
                return Err(format!(
                    "{NAME}: the stream header's extradata is {:?}, not this codec's version 1",
                    coded.extradata
                ))
            }
        };
        let CodedFormat::Video(video) = coded.format else {
            return Err(format!("{NAME} decodes video alone"));
        };
        // The header names the pixel format that was coded; one naming none
        // decodes as the most preferred unless the params name one. A packet
        // stating another is refused as it arrives.
        let mut pix_fmt = PIXEL_FORMATS[usize::from(coded_pix.unwrap_or(0))].to_string();
        for (key, value) in params_object(&params)? {
            match (key.as_str(), value.as_str()) {
                ("pix_fmt", Some(named)) if pix_code(named).is_some() => {
                    if coded_pix.is_some_and(|coded| pix_code(named) != Some(coded)) {
                        return Err(format!(
                            "{NAME}: the stream is coded {pix_fmt}, and the params ask for {named}"
                        ));
                    }
                    pix_fmt = named.to_string()
                }
                _ => return Err(format!("{NAME} decoder takes pix_fmt alone, got: {key}")),
            }
        }
        let pix = pix_code(&pix_fmt).expect("checked above");
        DECODING.with(|d| {
            *d.borrow_mut() = Some(Opened {
                width: video.width,
                height: video.height,
                pix,
            })
        });
        Ok(Format::Video(VideoFormat {
            width: video.width,
            height: video.height,
            pix_fmt,
            color: video.color,
        }))
    }

    fn decode(packets: Vec<Packet>, _last: bool) -> Result<Vec<RawFrame>, String> {
        let opened = DECODING
            .with(|d| *d.borrow())
            .ok_or_else(|| format!("{NAME}: decode before init"))?;
        let len = frame_len(opened);
        let mut frames = Vec::with_capacity(packets.len());
        for packet in packets {
            let data = &packet.data;
            if data.len() < HEADER_LEN || &data[..2] != MAGIC {
                return Err(format!(
                    "{NAME}: a packet at pts {} has no FT header",
                    packet.pts
                ));
            }
            let width = u16::from_le_bytes([data[2], data[3]]);
            let height = u16::from_le_bytes([data[4], data[5]]);
            let pix = data[6];
            if (u32::from(width), u32::from(height), pix)
                != (opened.width, opened.height, opened.pix)
            {
                return Err(format!(
                    "{NAME}: a packet at pts {} states {width}x{height} format {pix}, and the \
                     stream was opened for {}x{} format {}",
                    packet.pts, opened.width, opened.height, opened.pix
                ));
            }
            frames.push(RawFrame {
                pts: packet.pts,
                duration: packet.duration,
                data: expand(&data[HEADER_LEN..], len)?,
            });
        }
        Ok(frames)
    }
}

export!(TestCodec);
