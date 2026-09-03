//! NUT, the wire between the sidecar and the ffmpeg processes on either side.
//!
//! Raw frames cannot carry a timestamp; NUT can, ffmpeg reads and writes it
//! natively on a pipe, and it costs a few dozen bytes per frame. Only the
//! subset the compiler puts on those pipes is implemented: one stream, in NUT
//! version 3, carrying uncompressed video frames, interleaved pcm, or the
//! encoded packets of [`CODED_VIDEO_FOURCCS`] and [`CODED_AUDIO_FOURCCS`].
//! Anything else is refused by name rather than guessed at.
//!
//! What is read: the main header (version, stream count, time bases, the
//! framecode table, elision headers), the stream header (fourcc, the geometry
//! its class calls for, time base, pts coding, codec extradata), syncpoints,
//! and packets with their coded PTS and keyframe flag. Info and index packets
//! are skipped, as is any startcode this subset does not know.
//!
//! Payloads are opaque here whichever kind the stream is: a video packet is a
//! frame - encoded bytes or raw pixels - and an audio packet is however many
//! samples the producer chose, and none is looked inside.
//!
//! What is written: the same headers, one framecode that codes every field
//! explicitly, a syncpoint whenever the last one is further back than
//! `max_distance`, and frames carrying their PTS and a header checksum. A
//! raw stream's frames are always keyframes; an encoded stream's packets
//! state their own keyframe flag and must be handed over in decode order, so
//! a reader can work their dts back out the way it works a demuxed input's.
//!
//! # The annotation stream
//!
//! A sidecar that emits rows can put them on the wire beside the frames, so
//! the next sidecar reads both from one pipe. That is stream 1: codec tag
//! [`ANNOTATION_FOURCC`], stream class [`ANNOTATION_CLASS`], the media
//! stream's time base, one packet per frame that has rows, its payload the rows as NDJSON
//! and its PTS the frame's. A frame with no rows gets no packet, and the
//! packet is written before the frame it belongs to.
//!
//! Rows a module had no frame to put them on ride one further packet, after
//! every frame's: a single JSON object `{"trailing": [...]}` rather than
//! NDJSON, which is what tells the two apart. It comes last and there is at
//! most one.
//!
//! **This stream is private to this demuxer.** ffmpeg has no codec for the
//! tag and no reason to be handed it: the annotated form belongs on a
//! sidecar-to-sidecar pipe only, and both ends opt into it. Every
//! ffmpeg-facing edge stays the single-stream subset above, and a second
//! stream arriving there is still refused by name.

mod adts;
mod bytes;
mod demux;
mod mux;

pub use demux::Demuxer;
pub use mux::Muxer;

/// The 25 bytes every NUT file starts with.
pub const FILE_ID: &[u8] = b"nut/multimedia container\0";

/// The NUT version read and written. Version 4 adds per-frame side data and
/// broadcast timestamps, neither of which belongs on this wire.
pub const VERSION: u64 = 3;

/// How far apart syncpoints may be, in bytes. ffmpeg's own value; the format
/// caps it at 65536.
pub const MAX_DISTANCE: u64 = 32767;

/// The stream the annotation packets ride on, beside the video's stream 0.
pub const ANNOTATION_STREAM_ID: u64 = 1;

/// NUT's stream class for data that is neither video, audio nor subtitles.
pub const ANNOTATION_CLASS: u64 = 3;

/// The codec tag on the annotation stream. Private to this demuxer: no
/// ffmpeg build has a decoder for it, by design.
pub const ANNOTATION_FOURCC: &[u8; 4] = b"FRWD";

/// The only key of the record trailing rows ride in.
pub const TRAILING_KEY: &str = "trailing";

pub const MAIN_STARTCODE: u64 = 0x4E4D_7A56_1F5F_04AD;
pub const STREAM_STARTCODE: u64 = 0x4E53_1140_5BF2_F9DB;
pub const SYNCPOINT_STARTCODE: u64 = 0x4E4B_E4AD_EECA_4569;
pub const INDEX_STARTCODE: u64 = 0x4E58_DD67_2F23_E64E;
pub const INFO_STARTCODE: u64 = 0x4E49_AB68_B596_BA78;

/// Frame flags, as they appear in the framecode table and in a frame's own
/// `coded_flags`.
pub mod flags {
    pub const KEY: u64 = 1;
    pub const EOR: u64 = 2;
    pub const CODED_PTS: u64 = 8;
    pub const STREAM_ID: u64 = 16;
    pub const SIZE_MSB: u64 = 32;
    pub const CHECKSUM: u64 = 64;
    pub const RESERVED: u64 = 128;
    pub const SM_DATA: u64 = 256;
    pub const HEADER_IDX: u64 = 1024;
    pub const MATCH_TIME: u64 = 2048;
    pub const CODED: u64 = 4096;
    pub const INVALID: u64 = 8192;
}

/// The unit PTS are counted in, as a rational number of seconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeBase {
    pub num: u64,
    pub den: u64,
}

impl TimeBase {
    /// A timestamp in seconds.
    pub fn seconds(&self, pts: i64) -> f64 {
        pts as f64 * self.num as f64 / self.den as f64
    }
}

/// What one frame's own header said about it, past its bytes. For an encoded
/// stream a frame is a packet; the names differ, the wire does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Packet {
    /// Presentation timestamp, as the frame coded it. Frames arrive in
    /// decode order, so where the stream reorders this is not monotonic.
    pub pts: i64,
    /// Decode timestamp, worked out from the pts stream through a reorder
    /// buffer of `decode_delay + 1` entries, the way ffmpeg's own demuxing
    /// does. None for the first `decode_delay` frames, where the wire does
    /// not settle it. With no reordering it is the pts itself.
    pub dts: Option<i64>,
    /// The frame header's keyframe flag.
    pub keyframe: bool,
}

/// NUT's stream class for video, and for audio.
pub const VIDEO_CLASS: u64 = 0;
pub const AUDIO_CLASS: u64 = 1;

/// What a stream on this wire carries, past the fields both kinds share.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Media {
    Video {
        width: u32,
        height: u32,
        sample_width: u64,
        sample_height: u64,
        colorspace_type: u64,
    },
    Audio {
        sample_rate: u32,
        channels: u32,
    },
}

/// The one media stream this wire carries, as the stream header describes it.
/// The muxer writes an output header from the input's, so everything the
/// consumer needs survives the hop.
#[derive(Debug, Clone)]
pub struct Stream {
    /// The codec tag. `RGBA`, `I420`, the two pcm tags and the coded tags of
    /// [`CODED_VIDEO_FOURCCS`] and [`CODED_AUDIO_FOURCCS`] are what this
    /// subset knows.
    pub fourcc: Vec<u8>,
    pub time_base: TimeBase,
    /// How many low bits of a PTS a frame may code, instead of all of it.
    pub msb_pts_shift: u32,
    pub max_pts_distance: u64,
    /// How many packets the decoder holds back before the first picture
    /// leaves it. Zero for every raw stream; a coded stream with B-frames
    /// reorders, and this is by how much.
    pub decode_delay: u64,
    /// The codec's out-of-band header, from the stream header's
    /// codec-specific field: for h264, the SPS and PPS. Empty for raw
    /// streams.
    pub extradata: Vec<u8>,
    pub media: Media,
}

/// The pixel formats this wire carries, and the codec tag ffmpeg gives each.
const PIX_FMT_FOURCCS: &[(&str, &[u8; 4])] = &[("rgba", b"RGBA"), ("yuv420p", b"I420")];

/// The sample formats this wire carries, and the codec tag ffmpeg gives each:
/// `pcm_f32le` and `pcm_s16le`, both interleaved.
const SAMPLE_FMT_FOURCCS: &[(&str, &[u8; 4])] = &[("f32", b"PFD\x20"), ("s16", b"PSD\x10")];

/// The coded video streams this wire carries, and the tags ffmpeg gives each
/// in NUT (its muxer writes the first; a demuxer accepts the aliases too).
/// Payloads stay opaque: a packet is handed through exactly as it arrived.
pub const CODED_VIDEO_FOURCCS: &[(&str, &[&[u8; 4]])] = &[
    ("h264", &[b"H264", b"h264", b"avc1", b"AVC1"]),
    ("hevc", &[b"HEVC", b"hevc", b"hev1", b"hvc1"]),
    ("av1", &[b"AV01", b"av01"]),
];

/// The coded audio streams this wire carries, and the tags ffmpeg gives each
/// in NUT. Its muxer writes `ff 00 00 00` for AAC, and puts the codec's
/// AudioSpecificConfig in the stream header's extradata; `mp4a` is the tag
/// the same codec takes in mp4, accepted here as an alias. An aac stream
/// remuxed `-c copy` off ADTS (HLS, mpegts) carries no extradata at all - the
/// demuxer derives it from the first packet's ADTS header instead, and
/// strips that header from every packet; see [`adts`].
pub const CODED_AUDIO_FOURCCS: &[(&str, &[&[u8; 4]])] =
    &[("aac", &[b"\xff\x00\x00\x00", b"mp4a", b"MP4A"])];

impl Stream {
    /// A video stream of `pix_fmt` frames, for building a header from nothing
    /// but geometry.
    pub fn video(pix_fmt: &str, width: u32, height: u32, time_base: TimeBase) -> Option<Stream> {
        Some(Stream {
            fourcc: fourcc_for_pix_fmt(pix_fmt)?.to_vec(),
            time_base,
            msb_pts_shift: 14,
            max_pts_distance: time_base.den.div_ceil(time_base.num.max(1)),
            decode_delay: 0,
            extradata: Vec::new(),
            media: Media::Video {
                width,
                height,
                sample_width: 1,
                sample_height: 1,
                colorspace_type: 0,
            },
        })
    }

    /// An audio stream of interleaved `sample_fmt` samples. The time base is
    /// the natural one, where a tick is a sample.
    pub fn audio(sample_fmt: &str, sample_rate: u32, channels: u32) -> Option<Stream> {
        let time_base = TimeBase {
            num: 1,
            den: u64::from(sample_rate),
        };
        Some(Stream {
            fourcc: fourcc_for_sample_fmt(sample_fmt)?.to_vec(),
            time_base,
            msb_pts_shift: 14,
            max_pts_distance: u64::from(sample_rate),
            decode_delay: 0,
            extradata: Vec::new(),
            media: Media::Audio {
                sample_rate,
                channels,
            },
        })
    }

    /// The pixel format the codec tag names, or None for anything that is not
    /// a video stream this subset carries.
    pub fn pix_fmt(&self) -> Option<&'static str> {
        matches!(self.media, Media::Video { .. })
            .then(|| named(PIX_FMT_FOURCCS, &self.fourcc))
            .flatten()
    }

    /// ffmpeg's name for the coded codec the tag names, from the table for
    /// this stream's own kind. None for a raw stream and for any codec this
    /// wire does not carry.
    pub fn codec_name(&self) -> Option<&'static str> {
        let table = match self.media {
            Media::Video { .. } => CODED_VIDEO_FOURCCS,
            Media::Audio { .. } => CODED_AUDIO_FOURCCS,
        };
        table
            .iter()
            .find(|(_, tags)| tags.iter().any(|tag| self.fourcc == tag.as_slice()))
            .map(|(name, _)| *name)
    }

    /// The sample format the codec tag names, or None for anything that is not
    /// an audio stream this subset carries.
    pub fn sample_fmt(&self) -> Option<&'static str> {
        matches!(self.media, Media::Audio { .. })
            .then(|| named(SAMPLE_FMT_FOURCCS, &self.fourcc))
            .flatten()
    }

    /// The frame geometry, for a video stream.
    pub fn video_geometry(&self) -> Option<(u32, u32)> {
        match self.media {
            Media::Video { width, height, .. } => Some((width, height)),
            Media::Audio { .. } => None,
        }
    }

    /// The rate and channel count, for an audio stream.
    pub fn audio_geometry(&self) -> Option<(u32, u32)> {
        match self.media {
            Media::Audio {
                sample_rate,
                channels,
            } => Some((sample_rate, channels)),
            Media::Video { .. } => None,
        }
    }

    /// `video` or `audio`, for a message.
    pub fn kind(&self) -> &'static str {
        match self.media {
            Media::Video { .. } => "video",
            Media::Audio { .. } => "audio",
        }
    }

    /// The codec tag as something safe to put in an error message.
    pub fn fourcc_name(&self) -> String {
        String::from_utf8_lossy(&self.fourcc)
            .chars()
            .map(|c| {
                if c.is_ascii_graphic() || c == ' ' {
                    c
                } else {
                    '?'
                }
            })
            .collect()
    }
}

/// The name a table gives a codec tag.
fn named(table: &[(&'static str, &[u8; 4])], fourcc: &[u8]) -> Option<&'static str> {
    table
        .iter()
        .find(|(_, tag)| fourcc == tag.as_slice())
        .map(|(name, _)| *name)
}

/// The codec tag ffmpeg gives `pix_fmt` in NUT.
pub fn fourcc_for_pix_fmt(pix_fmt: &str) -> Option<&'static [u8; 4]> {
    PIX_FMT_FOURCCS
        .iter()
        .find(|(name, _)| *name == pix_fmt)
        .map(|(_, tag)| *tag)
}

/// The codec tag ffmpeg gives the pcm codec of `sample_fmt` in NUT.
pub fn fourcc_for_sample_fmt(sample_fmt: &str) -> Option<&'static [u8; 4]> {
    SAMPLE_FMT_FOURCCS
        .iter()
        .find(|(name, _)| *name == sample_fmt)
        .map(|(_, tag)| *tag)
}

/// The codec tag ffmpeg gives `codec` in NUT, from the table for `kind`
/// (`"video"` or `"audio"`) - the muxer's own tag, the first of the aliases
/// a demuxer also accepts. None for a codec this wire does not carry, or a
/// `kind` that names neither.
pub fn fourcc_for_coded(kind: &str, codec: &str) -> Option<&'static [u8; 4]> {
    let table = match kind {
        "video" => CODED_VIDEO_FOURCCS,
        "audio" => CODED_AUDIO_FOURCCS,
        _ => return None,
    };
    table
        .iter()
        .find(|(name, _)| *name == codec)
        .map(|(_, tags)| tags[0])
}

/// The pixel formats this wire carries, most common first.
pub fn supported_pix_fmts() -> Vec<&'static str> {
    PIX_FMT_FOURCCS.iter().map(|(name, _)| *name).collect()
}

/// The sample formats this wire carries, most common first.
pub fn supported_sample_fmts() -> Vec<&'static str> {
    SAMPLE_FMT_FOURCCS.iter().map(|(name, _)| *name).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn a_stream() -> Stream {
        Stream::video("rgba", 8, 8, TimeBase { num: 1, den: 65536 }).expect("rgba is carried")
    }

    /// Writes `frames` as NUT and reads them back through the demuxer.
    fn round_trip(stream: &Stream, frames: &[(i64, Vec<u8>)]) -> Vec<(i64, Vec<u8>)> {
        let mut wire = Vec::new();
        {
            let mut muxer = Muxer::new(&mut wire, stream).expect("write headers");
            for (pts, data) in frames {
                muxer.write_frame(*pts, data).expect("write frame");
            }
            muxer.finish().expect("finish");
        }
        let mut demuxer = Demuxer::open(Cursor::new(wire)).expect("read headers");
        assert_eq!(demuxer.stream().media, stream.media);
        assert_eq!(demuxer.stream().time_base, stream.time_base);
        assert_eq!(demuxer.stream().fourcc, stream.fourcc);

        let mut out = Vec::new();
        let mut buf = Vec::new();
        while let Some(pts) = demuxer.read_frame(&mut buf).expect("read frame") {
            out.push((pts, buf.clone()));
        }
        out
    }

    #[test]
    fn frames_and_their_timestamps_survive_a_round_trip() {
        let stream = a_stream();
        let frames: Vec<(i64, Vec<u8>)> = (0..5i64)
            .map(|i| (i * 65536, vec![i as u8; 8 * 8 * 4]))
            .collect();
        assert_eq!(round_trip(&stream, &frames), frames);
    }

    #[test]
    fn colorspace_and_aspect_ratio_survive_a_round_trip() {
        // What an upstream sidecar's header declared: Rec 709 full range,
        // anamorphic 4:3 samples. The output header is built from the
        // input's, so both must come back exactly - `round_trip` compares
        // the whole `media`.
        let mut stream = a_stream();
        stream.media = Media::Video {
            width: 8,
            height: 8,
            sample_width: 4,
            sample_height: 3,
            colorspace_type: 18,
        };
        let frames: Vec<(i64, Vec<u8>)> = (0..2i64)
            .map(|i| (i * 65536, vec![i as u8; 8 * 8 * 4]))
            .collect();
        assert_eq!(round_trip(&stream, &frames), frames);
    }

    #[test]
    fn timestamps_need_not_be_evenly_spaced() {
        let stream = a_stream();
        let frames: Vec<(i64, Vec<u8>)> = [0i64, 7, 1_000_000, 1_000_001]
            .iter()
            .map(|pts| (*pts, vec![0xAB; 8 * 8 * 4]))
            .collect();
        assert_eq!(round_trip(&stream, &frames), frames);
    }

    #[test]
    fn frames_larger_than_the_syncpoint_distance_round_trip() {
        let stream =
            Stream::video("rgba", 128, 128, TimeBase { num: 1, den: 25 }).expect("rgba is carried");
        let frames: Vec<(i64, Vec<u8>)> = (0..3i64)
            .map(|i| (i, vec![i as u8; 128 * 128 * 4]))
            .collect();
        assert_eq!(round_trip(&stream, &frames), frames);
    }

    /// Writes `frames` with their rows on the annotation stream and reads
    /// both back, pairing each frame's rows with it by PTS.
    fn round_trip_annotated(
        stream: &Stream,
        frames: &[(i64, Vec<u8>, Vec<String>)],
    ) -> Vec<(i64, Vec<u8>, Vec<String>)> {
        let mut wire = Vec::new();
        {
            let mut muxer = Muxer::with_annotations(&mut wire, stream).expect("write headers");
            for (pts, data, rows) in frames {
                muxer.write_rows(*pts, rows).expect("write rows");
                muxer.write_frame(*pts, data).expect("write frame");
            }
            muxer.finish().expect("finish");
        }
        let mut demuxer = Demuxer::open_annotated(Cursor::new(wire)).expect("read headers");
        assert!(demuxer.has_annotations(), "the second stream was declared");

        let mut out = Vec::new();
        let mut buf = Vec::new();
        while let Some(pts) = demuxer.read_frame(&mut buf).expect("read frame") {
            out.push((pts, buf.clone(), demuxer.take_rows(pts)));
        }
        out
    }

    #[test]
    fn rows_come_back_attached_to_their_own_frame() {
        let stream = a_stream();
        let frame = |i: u8| vec![i; 8 * 8 * 4];
        let frames = vec![
            (
                0i64,
                frame(0),
                vec![r#"{"x":1,"y":2,"w":3,"h":4}"#.to_string()],
            ),
            // No rows: this frame gets no packet at all, and must still come
            // back with an empty list rather than its neighbour's rows.
            (65536, frame(1), vec![]),
            (
                131_072,
                frame(2),
                vec![
                    r#"{"x":5,"y":6,"w":7,"h":8}"#.to_string(),
                    r#"{"x":9,"y":10,"w":11,"h":12}"#.to_string(),
                ],
            ),
        ];
        assert_eq!(round_trip_annotated(&stream, &frames), frames);
    }

    #[test]
    fn the_trailing_record_comes_back_belonging_to_no_frame() {
        let stream = a_stream();
        let trailing = vec![r#"{"frames":2}"#.to_string(), r#"{"cuts":1}"#.to_string()];
        let mut wire = Vec::new();
        {
            let mut muxer = Muxer::with_annotations(&mut wire, &stream).expect("write headers");
            for pts in [0i64, 65536] {
                muxer
                    .write_rows(pts, &[r#"{"x":1,"y":2,"w":3,"h":4}"#.to_string()])
                    .expect("write rows");
                muxer
                    .write_frame(pts, &vec![7u8; 8 * 8 * 4])
                    .expect("write frame");
            }
            muxer
                .write_trailing(65536, &trailing)
                .expect("write the trailing record");
            muxer.finish().expect("finish");
        }

        let mut demuxer = Demuxer::open_annotated(Cursor::new(wire)).expect("read headers");
        let mut buf = Vec::new();
        while let Some(pts) = demuxer.read_frame(&mut buf).expect("read frame") {
            assert_eq!(
                demuxer.take_rows(pts).len(),
                1,
                "the record is no frame's rows"
            );
        }
        assert_eq!(
            demuxer.take_trailing(),
            trailing,
            "the rows come back as they were written"
        );
    }

    #[test]
    fn a_trailing_row_that_is_not_json_has_no_record_to_ride_in() {
        let stream = a_stream();
        let mut wire = Vec::new();
        let mut muxer = Muxer::with_annotations(&mut wire, &stream).expect("write headers");
        let err = muxer
            .write_trailing(0, &["not json".to_string()])
            .unwrap_err()
            .to_string();
        assert!(err.contains("not JSON"), "{err}");
    }

    #[test]
    fn an_annotated_stream_read_as_a_plain_one_is_refused() {
        let stream = a_stream();
        let mut wire = Vec::new();
        {
            let mut muxer = Muxer::with_annotations(&mut wire, &stream).expect("write headers");
            muxer
                .write_rows(0, &[r#"{"x":0,"y":0,"w":1,"h":1}"#.to_string()])
                .expect("write rows");
            muxer.write_frame(0, &vec![0u8; 8 * 8 * 4]).expect("frame");
            muxer.finish().expect("finish");
        }
        let err = match Demuxer::open(Cursor::new(wire)) {
            Ok(_) => panic!("the ffmpeg-facing wire carries one stream"),
            Err(e) => e.to_string(),
        };
        assert_eq!(err, "NUT input carries 2 streams; this wire carries one");
    }

    #[test]
    fn a_plain_stream_is_still_read_when_annotations_are_asked_for() {
        let stream = a_stream();
        let mut wire = Vec::new();
        {
            let mut muxer = Muxer::new(&mut wire, &stream).expect("write headers");
            muxer.write_frame(7, &vec![3u8; 8 * 8 * 4]).expect("frame");
            muxer.finish().expect("finish");
        }
        let mut demuxer = Demuxer::open_annotated(Cursor::new(wire)).expect("read headers");
        assert!(!demuxer.has_annotations());
        let mut buf = Vec::new();
        assert_eq!(demuxer.read_frame(&mut buf).expect("read frame"), Some(7));
        assert!(demuxer.take_rows(7).is_empty());
    }

    #[test]
    fn a_codec_tag_this_wire_does_not_carry_is_named() {
        let mut stream = a_stream();
        stream.fourcc = b"XYZW".to_vec();
        assert_eq!(stream.pix_fmt(), None);
        assert_eq!(stream.fourcc_name(), "XYZW");
    }

    /// An encoded AAC stream header, as ffmpeg's NUT muxer writes one: the
    /// codec tag `ff 00 00 00`, and the AudioSpecificConfig as extradata.
    fn an_aac_stream() -> Stream {
        Stream {
            fourcc: b"\xff\x00\x00\x00".to_vec(),
            time_base: TimeBase { num: 1, den: 48000 },
            msb_pts_shift: 14,
            max_pts_distance: 48000,
            decode_delay: 0,
            // 48 kHz mono AAC-LC, as ffprobe reported it.
            extradata: vec![0x11, 0x88, 0x56, 0xe5, 0x00],
            media: Media::Audio {
                sample_rate: 48000,
                channels: 1,
            },
        }
    }

    #[test]
    fn an_encoded_audio_stream_is_named_by_its_tag() {
        let stream = an_aac_stream();
        assert_eq!(stream.codec_name(), Some("aac"));
        assert_eq!(stream.sample_fmt(), None);
        assert_eq!(stream.audio_geometry(), Some((48000, 1)));
    }

    #[test]
    fn an_audio_tag_this_wire_does_not_carry_names_no_codec() {
        let mut stream = an_aac_stream();
        stream.fourcc = b"OPUS".to_vec();
        assert_eq!(stream.codec_name(), None);
        assert_eq!(stream.fourcc_name(), "OPUS");
    }

    #[test]
    fn a_video_tag_never_names_an_audio_codec() {
        let mut stream = an_aac_stream();
        stream.fourcc = b"H264".to_vec();
        assert_eq!(stream.codec_name(), None);
    }

    #[test]
    fn encoded_audio_packets_and_their_extradata_survive_a_round_trip() {
        let stream = an_aac_stream();
        // Each packet is one AAC frame of 1024 samples, so the timestamps
        // step by 1024 in the stream's own ticks.
        let frames: Vec<(i64, Vec<u8>)> = (0..4i64)
            .map(|i| (i * 1024, vec![0x21, i as u8, 0x10, 0x04]))
            .collect();
        let mut wire = Vec::new();
        {
            let mut muxer = Muxer::new(&mut wire, &stream).expect("write headers");
            for (pts, data) in &frames {
                let packet = Packet {
                    pts: *pts,
                    dts: None,
                    keyframe: true,
                };
                muxer.write_coded(&packet, data).expect("write packet");
            }
            muxer.finish().expect("finish");
        }
        let mut demuxer = Demuxer::open(Cursor::new(wire)).expect("read headers");
        assert_eq!(demuxer.stream().codec_name(), Some("aac"));
        assert_eq!(demuxer.stream().extradata, stream.extradata);
        let mut out = Vec::new();
        let mut buf = Vec::new();
        while let Some(pts) = demuxer.read_frame(&mut buf).expect("read packet") {
            out.push((pts, buf.clone()));
        }
        assert_eq!(out, frames);
    }

    #[test]
    fn yuv420p_uses_the_tag_ffmpeg_writes() {
        assert_eq!(fourcc_for_pix_fmt("yuv420p"), Some(b"I420"));
        assert_eq!(fourcc_for_pix_fmt("rgba"), Some(b"RGBA"));
        assert_eq!(fourcc_for_pix_fmt("gray"), None);
    }

    #[test]
    fn fourcc_for_coded_reads_the_muxers_own_tag() {
        assert_eq!(fourcc_for_coded("video", "h264"), Some(b"H264"));
        assert_eq!(fourcc_for_coded("video", "hevc"), Some(b"HEVC"));
        assert_eq!(fourcc_for_coded("video", "av1"), Some(b"AV01"));
        assert_eq!(fourcc_for_coded("audio", "aac"), Some(b"\xff\x00\x00\x00"));
        assert_eq!(fourcc_for_coded("video", "aac"), None, "wrong kind");
        assert_eq!(fourcc_for_coded("video", "vp9"), None, "not carried");
        assert_eq!(fourcc_for_coded("subtitle", "h264"), None, "not a kind");
    }
}
