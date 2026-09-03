//! Writing NUT: the headers that describe the stream, and frames that carry
//! their PTS. What comes out is what real ffmpeg reads with `-f nut`.

use std::io::Write;

use anyhow::{anyhow, bail, Result};

use super::bytes::{crc32, put_s, put_u32, put_u64, put_v, put_vb};
use super::{
    flags, Media, Packet, Stream, ANNOTATION_CLASS, ANNOTATION_FOURCC, ANNOTATION_STREAM_ID,
    AUDIO_CLASS, FILE_ID, MAIN_STARTCODE, MAX_DISTANCE, STREAM_STARTCODE, SYNCPOINT_STARTCODE,
    TRAILING_KEY, VERSION, VIDEO_CLASS,
};

/// The framecode every frame uses. It sets only `CODED`, so each frame states
/// the rest of its flags itself: one table entry serves every frame, and the
/// table needs no tuning to the stream - a raw frame's fixed flags and a
/// coded packet's per-packet flags both ride through it unchanged.
const EXPLICIT_FRAME_CODE: u8 = 1;

/// What each raw frame states. Uncompressed frames are all keyframes; the PTS
/// and the size are always coded, and the header checksum keeps the frame
/// readable however far it sits from a syncpoint.
const FRAME_FLAGS: u64 = flags::KEY | flags::CODED_PTS | flags::SIZE_MSB | flags::CHECKSUM;

/// What an encoded packet states, past its own keyframe flag: the PTS and the
/// size are always coded, and the header checksum keeps the frame readable
/// however far it sits from a syncpoint. `write_coded` adds `flags::KEY`
/// itself, per packet, since only the packet knows.
const CODED_FRAME_FLAGS: u64 = flags::CODED_PTS | flags::SIZE_MSB | flags::CHECKSUM;

/// The framecode table's size multiplier. With it at 1 a frame's coded size
/// is its byte count.
const SIZE_MUL: u64 = 1;

/// A packet body this wide would need a checksum over its own header. The
/// headers written here are tens of bytes, so it is a guard, not a case.
const HEADER_CHECKSUM_THRESHOLD: usize = 4096;

/// Writes one NUT stream: headers from a [`Stream`], then frames. With
/// annotations on it writes a second stream beside the video, for the rows a
/// module emitted; see this module's documentation for who may read it.
pub struct Muxer<W> {
    out: W,
    stream: Stream,
    annotations: bool,
    pos: u64,
    last_syncpoint: Option<u64>,
}

impl<W: Write> Muxer<W> {
    /// Writes the identifier and both headers, so the reader on the far side
    /// knows the geometry before the first frame arrives.
    pub fn new(out: W, stream: &Stream) -> Result<Muxer<W>> {
        Muxer::open(out, stream, false)
    }

    /// `new`, plus the annotation stream. Only another sidecar reads what
    /// this writes.
    pub fn with_annotations(out: W, stream: &Stream) -> Result<Muxer<W>> {
        Muxer::open(out, stream, true)
    }

    fn open(out: W, stream: &Stream, annotations: bool) -> Result<Muxer<W>> {
        if stream.msb_pts_shift >= 63 {
            bail!(
                "NUT output would shift PTS by {} bits",
                stream.msb_pts_shift
            );
        }
        let mut muxer = Muxer {
            out,
            stream: stream.clone(),
            annotations,
            pos: 0,
            last_syncpoint: None,
        };
        muxer.write_bytes(FILE_ID)?;
        let main = main_header(&muxer.stream, annotations);
        muxer.write_packet(MAIN_STARTCODE, &main)?;
        let header = stream_header(&muxer.stream);
        muxer.write_packet(STREAM_STARTCODE, &header)?;
        if annotations {
            let header = annotation_stream_header(&muxer.stream);
            muxer.write_packet(STREAM_STARTCODE, &header)?;
        }
        Ok(muxer)
    }

    /// One frame, at `pts` in the stream's time base. Refused on an encoded
    /// stream: an encoded packet states its own keyframe flag, which
    /// `write_coded` carries and this fixed-flags path does not.
    pub fn write_frame(&mut self, pts: i64, data: &[u8]) -> Result<()> {
        if self.stream.codec_name().is_some() {
            bail!(
                "write_frame on an encoded {} stream; use write_coded",
                self.stream.fourcc_name()
            );
        }
        self.write_packet_for(0, pts, FRAME_FLAGS, data)
    }

    /// One encoded packet, in the decode order it must be handed to this
    /// method in. Refused on a raw stream: use `write_frame` there instead.
    ///
    /// `packet.pts` is written as `write_frame` writes a raw frame's; NUT
    /// carries no dts field at all, so `packet.dts` is not written, and a
    /// reader - this crate's own `Demuxer` included - works dts back out of
    /// the pts values it receives, in the order it receives them, through a
    /// reorder buffer of `decode_delay + 1` entries. That is also why the
    /// packets must arrive here in decode order: the buffer that recovers
    /// dts on the way out depends on it. `packet.keyframe` sets `flags::KEY`
    /// on this packet alone, which is what lets key and non-key packets share
    /// the wire. Nothing here carries a duration: what NUT implies from the
    /// next packet's pts is all a reader gets back.
    pub fn write_coded(&mut self, packet: &Packet, data: &[u8]) -> Result<()> {
        if self.stream.codec_name().is_none() {
            bail!(
                "write_coded on a raw {} stream; use write_frame",
                self.stream.fourcc_name()
            );
        }
        let frame_flags = CODED_FRAME_FLAGS | if packet.keyframe { flags::KEY } else { 0 };
        self.write_packet_for(0, packet.pts, frame_flags, data)
    }

    /// The rows a module produced for the frame at `pts`, as NDJSON, on the
    /// annotation stream. Nothing is written for a frame with no rows, and
    /// the packet precedes its frame so a reader has it in hand when the
    /// frame arrives.
    pub fn write_rows(&mut self, pts: i64, rows: &[String]) -> Result<()> {
        if !self.annotations {
            bail!("NUT output carries no annotation stream, so rows have nowhere to go");
        }
        if rows.is_empty() {
            return Ok(());
        }
        self.write_packet_for(
            ANNOTATION_STREAM_ID,
            pts,
            FRAME_FLAGS,
            rows.join("\n").as_bytes(),
        )
    }

    /// The rows a module had no frame to put them on, as one record after
    /// every frame's. `pts` is where the stream got to, which is the last
    /// frame's timestamp.
    pub fn write_trailing(&mut self, pts: i64, rows: &[String]) -> Result<()> {
        if !self.annotations {
            bail!("NUT output carries no annotation stream, so trailing rows have nowhere to go");
        }
        if rows.is_empty() {
            return Ok(());
        }
        let record = trailing_record(rows)?;
        self.write_packet_for(ANNOTATION_STREAM_ID, pts, FRAME_FLAGS, record.as_bytes())
    }

    /// One frame on `stream_id`, at `pts` in the stream's time base, stating
    /// `frame_flags` - the fixed flags of a raw frame, or the per-packet
    /// flags `write_coded` works out from the packet's own keyframe bit. The
    /// framecode table's one entry states `CODED` for every frame, so
    /// whichever flags are passed in are what a reader gets back out.
    fn write_packet_for(
        &mut self,
        stream_id: u64,
        pts: i64,
        frame_flags: u64,
        data: &[u8],
    ) -> Result<()> {
        if pts < 0 {
            bail!("NUT output cannot carry the negative PTS {pts}");
        }
        if self
            .last_syncpoint
            .is_none_or(|prev| self.pos - prev >= MAX_DISTANCE)
        {
            self.write_syncpoint(pts)?;
        }

        // The whole PTS, offset past the range reserved for frames coding
        // only its low bits. Absolute means a frame never depends on how far
        // the reader has got, which is what keeps a filtered stream's
        // timestamps exactly the ones that came in.
        let coded_pts = (pts as u64) + (1u64 << self.stream.msb_pts_shift);
        // The framecode table's entry names stream 0, so only the annotation
        // stream states an id of its own.
        let frame_flags = if stream_id == 0 {
            frame_flags
        } else {
            frame_flags | flags::STREAM_ID
        };

        let mut header = Vec::with_capacity(16);
        header.push(EXPLICIT_FRAME_CODE);
        put_v(&mut header, frame_flags ^ flags::CODED);
        if stream_id != 0 {
            put_v(&mut header, stream_id);
        }
        put_v(&mut header, coded_pts);
        // The table's size multiplier is one, so a frame's coded size is its
        // byte count.
        put_v(&mut header, data.len() as u64);
        let checksum = crc32(&header);
        put_u32(&mut header, checksum);

        self.write_bytes(&header)?;
        self.write_bytes(data)
    }

    /// Flushes everything written so far.
    pub fn finish(&mut self) -> Result<()> {
        self.out.flush()?;
        Ok(())
    }

    /// A syncpoint states where the stream has got to, and how far back the
    /// previous one was. For a raw stream every frame is a keyframe, so the
    /// last one before it is this syncpoint itself; an encoded stream places
    /// one wherever `MAX_DISTANCE` is crossed regardless of which packet that
    /// lands on, since nothing on this wire ever demuxes by seeking to one.
    fn write_syncpoint(&mut self, pts: i64) -> Result<()> {
        let here = self.pos;
        let back_ptr = self.last_syncpoint.map_or(0, |prev| (here - prev) / 16);
        let mut body = Vec::with_capacity(8);
        // One time base is declared, so its index adds nothing to the
        // timestamp.
        put_v(&mut body, pts as u64);
        put_v(&mut body, back_ptr);
        self.write_packet(SYNCPOINT_STARTCODE, &body)?;
        self.last_syncpoint = Some(here);
        Ok(())
    }

    /// A startcode, the length of what follows, and the body with its
    /// checksum appended.
    fn write_packet(&mut self, startcode: u64, fields: &[u8]) -> Result<()> {
        let mut body = fields.to_vec();
        put_u32(&mut body, crc32(fields));
        if body.len() > HEADER_CHECKSUM_THRESHOLD {
            bail!(
                "NUT output packet is {} bytes, past the {HEADER_CHECKSUM_THRESHOLD} at which a \
                 packet must carry a checksum over its own header",
                body.len()
            );
        }
        let mut packet = Vec::with_capacity(body.len() + 16);
        put_u64(&mut packet, startcode);
        put_v(&mut packet, body.len() as u64);
        packet.extend_from_slice(&body);
        self.write_bytes(&packet)
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        self.out.write_all(bytes)?;
        self.pos += bytes.len() as u64;
        Ok(())
    }
}

/// The rows written verbatim into the one record trailing rows ride in. Each
/// is checked to be JSON first, so what goes on the wire is one well-formed
/// object rather than a record a reader cannot parse.
fn trailing_record(rows: &[String]) -> Result<String> {
    let mut record = format!("{{\"{TRAILING_KEY}\":[");
    for (index, row) in rows.iter().enumerate() {
        let row = row.trim();
        serde_json::from_str::<serde::de::IgnoredAny>(row).map_err(|e| {
            anyhow!("trailing row {index} is not JSON, so it cannot ride in one record: {e}")
        })?;
        if index > 0 {
            record.push(',');
        }
        record.push_str(row);
    }
    record.push_str("]}");
    Ok(record)
}

/// The main header: the streams, one time base, and a framecode table with a
/// single usable entry.
fn main_header(stream: &Stream, annotations: bool) -> Vec<u8> {
    let mut body = Vec::with_capacity(64);
    put_v(&mut body, VERSION);
    put_v(&mut body, if annotations { 2 } else { 1 }); // stream count
    put_v(&mut body, MAX_DISTANCE);
    put_v(&mut body, 1); // time base count
    put_v(&mut body, stream.time_base.num);
    put_v(&mut body, stream.time_base.den);

    // Index 0, then index 1 - the one `EXPLICIT_FRAME_CODE` names - then
    // everything above it. A group's count leaves out index `N`, which is
    // reserved and always invalid: the last group covers 254 entries and
    // counts 253 of them.
    put_frame_code_group(&mut body, flags::INVALID, 0, 1);
    put_frame_code_group(&mut body, flags::CODED, SIZE_MUL, 1);
    put_frame_code_group(&mut body, flags::INVALID, 0, 253);

    put_v(&mut body, 0); // no elision headers
    body
}

/// One run of framecode table entries, with every field stated.
fn put_frame_code_group(body: &mut Vec<u8>, code_flags: u64, size_mul: u64, count: u64) {
    put_v(body, code_flags);
    put_v(body, 6); // fields stated: pts delta, size multiplier, stream, size, reserved, count
    put_s(body, 0); // pts delta, unused since every frame codes its own
    put_v(body, size_mul);
    put_v(body, 0); // stream
    put_v(body, 0); // size lsb
    put_v(body, 0); // reserved fields
    put_v(body, count);
}

/// The stream header: the codec tag, the geometry its class calls for, and
/// how PTS are coded.
fn stream_header(stream: &Stream) -> Vec<u8> {
    let class = match stream.media {
        Media::Video { .. } => VIDEO_CLASS,
        Media::Audio { .. } => AUDIO_CLASS,
    };
    let mut body = Vec::with_capacity(32);
    put_v(&mut body, 0); // stream id
    put_v(&mut body, class);
    put_vb(&mut body, &stream.fourcc);
    put_v(&mut body, 0); // time base id
    put_v(&mut body, u64::from(stream.msb_pts_shift));
    put_v(&mut body, stream.max_pts_distance);
    put_v(&mut body, stream.decode_delay);
    put_v(&mut body, 0); // stream flags
    put_vb(&mut body, &stream.extradata); // empty for every raw stream
    match stream.media {
        Media::Video {
            width,
            height,
            sample_width,
            sample_height,
            colorspace_type,
        } => {
            put_v(&mut body, u64::from(width));
            put_v(&mut body, u64::from(height));
            put_v(&mut body, sample_width);
            put_v(&mut body, sample_height);
            put_v(&mut body, colorspace_type);
        }
        Media::Audio {
            sample_rate,
            channels,
        } => {
            put_v(&mut body, u64::from(sample_rate)); // sample rate numerator
            put_v(&mut body, 1); // sample rate denominator
            put_v(&mut body, u64::from(channels));
        }
    }
    body
}

/// The annotation stream's header: the same time base and PTS coding as the
/// video, so a row packet's timestamp compares directly with a frame's, and a
/// data class, which carries no geometry.
fn annotation_stream_header(stream: &Stream) -> Vec<u8> {
    let mut body = Vec::with_capacity(24);
    put_v(&mut body, ANNOTATION_STREAM_ID);
    put_v(&mut body, ANNOTATION_CLASS);
    put_vb(&mut body, ANNOTATION_FOURCC);
    put_v(&mut body, 0); // time base id: the video's
    put_v(&mut body, u64::from(stream.msb_pts_shift));
    put_v(&mut body, stream.max_pts_distance);
    put_v(&mut body, 0); // decode delay
    put_v(&mut body, 0); // stream flags
    put_vb(&mut body, &[]); // no codec specific data
    body
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nut::{Demuxer, TimeBase, FILE_ID};

    fn a_stream() -> Stream {
        Stream::video("rgba", 2, 2, TimeBase { num: 1, den: 25 }).expect("rgba is carried")
    }

    /// An h264 stream header shaped like a real encoded one: SPS/PPS as
    /// extradata, and a decode delay of two - a reorder buffer of three pts -
    /// which is what a B-frame GOP needs.
    fn a_coded_h264_stream() -> Stream {
        Stream {
            fourcc: b"H264".to_vec(),
            time_base: TimeBase { num: 1, den: 25 },
            msb_pts_shift: 14,
            max_pts_distance: 25,
            decode_delay: 2,
            extradata: vec![0x67, 0x42, 0x00, 0x1e],
            media: Media::Video {
                width: 16,
                height: 16,
                sample_width: 1,
                sample_height: 1,
                colorspace_type: 0,
            },
        }
    }

    /// An AAC stream header: ffmpeg's own codec tag for it, and no
    /// reordering.
    fn a_coded_aac_stream() -> Stream {
        Stream {
            fourcc: b"\xff\x00\x00\x00".to_vec(),
            time_base: TimeBase { num: 1, den: 48000 },
            msb_pts_shift: 14,
            max_pts_distance: 48000,
            decode_delay: 0,
            extradata: vec![0x11, 0x88, 0x56, 0xe5, 0x00],
            media: Media::Audio {
                sample_rate: 48000,
                channels: 1,
            },
        }
    }

    fn wire_with(frames: &[(i64, Vec<u8>)]) -> Vec<u8> {
        let mut wire = Vec::new();
        {
            let mut muxer = Muxer::new(&mut wire, &a_stream()).expect("write headers");
            for (pts, data) in frames {
                muxer.write_frame(*pts, data).expect("write frame");
            }
            muxer.finish().expect("finish");
        }
        wire
    }

    #[test]
    fn the_stream_starts_with_the_identifier_and_the_main_startcode() {
        let wire = wire_with(&[]);
        assert_eq!(&wire[..FILE_ID.len()], FILE_ID);
        assert_eq!(
            u64::from_be_bytes(wire[FILE_ID.len()..FILE_ID.len() + 8].try_into().unwrap()),
            MAIN_STARTCODE
        );
    }

    #[test]
    fn a_syncpoint_precedes_the_first_frame() {
        let wire = wire_with(&[(0, vec![0u8; 16])]);
        let found = wire
            .windows(8)
            .position(|w| u64::from_be_bytes(w.try_into().unwrap()) == SYNCPOINT_STARTCODE);
        assert!(
            found.is_some(),
            "the first frame needs a syncpoint before it"
        );
    }

    #[test]
    fn syncpoints_come_no_further_apart_than_the_maximum_distance() {
        // Frames well under the distance, so several share one syncpoint.
        let frames: Vec<(i64, Vec<u8>)> = (0..64i64).map(|i| (i, vec![0u8; 1024])).collect();
        let wire = wire_with(&frames);
        let mut previous: Option<usize> = None;
        for start in 0..wire.len().saturating_sub(8) {
            let code = u64::from_be_bytes(wire[start..start + 8].try_into().unwrap());
            if code != SYNCPOINT_STARTCODE {
                continue;
            }
            if let Some(prev) = previous {
                assert!(
                    start - prev <= MAX_DISTANCE as usize + 2048,
                    "syncpoints {prev} and {start} are too far apart"
                );
            }
            previous = Some(start);
        }
        assert!(previous.is_some(), "there should be syncpoints");
    }

    #[test]
    fn a_negative_timestamp_is_refused() {
        let mut wire = Vec::new();
        let mut muxer = Muxer::new(&mut wire, &a_stream()).expect("write headers");
        let err = muxer.write_frame(-1, &[0u8; 16]).unwrap_err().to_string();
        assert!(err.contains("negative PTS"), "{err}");
    }

    #[test]
    fn write_coded_is_refused_on_a_raw_stream() {
        let mut wire = Vec::new();
        let mut muxer = Muxer::new(&mut wire, &a_stream()).expect("write headers");
        let packet = Packet {
            pts: 0,
            dts: None,
            keyframe: true,
        };
        let err = muxer
            .write_coded(&packet, &[0u8; 4])
            .unwrap_err()
            .to_string();
        assert!(err.contains("raw") && err.contains("write_frame"), "{err}");
    }

    #[test]
    fn write_frame_is_refused_on_a_coded_stream() {
        let mut wire = Vec::new();
        let mut muxer = Muxer::new(&mut wire, &a_coded_h264_stream()).expect("write headers");
        let err = muxer.write_frame(0, &[0u8; 4]).unwrap_err().to_string();
        assert!(
            err.contains("encoded") && err.contains("write_coded"),
            "{err}"
        );
    }

    #[test]
    fn coded_packets_round_trip_through_the_decode_order_reorder_buffer() {
        let stream = a_coded_h264_stream();
        // Decode order: I0 P3 B1 B2 P6 B4 B5, the shape one B-frame GOP
        // takes. Only the leading I frame is a keyframe.
        let packets: Vec<(i64, bool)> = vec![
            (0, true),
            (3, false),
            (1, false),
            (2, false),
            (6, false),
            (4, false),
            (5, false),
        ];
        // Worked out by hand from `decode_dts`'s reorder buffer of
        // `decode_delay + 1` entries: None until the buffer fills, then the
        // smallest pts seen so far that has not already come out.
        let expected_dts: [Option<i64>; 7] = [
            None,
            None,
            Some(0),
            Some(1),
            Some(2),
            Some(3),
            Some(4),
        ];

        let mut wire = Vec::new();
        {
            let mut muxer = Muxer::new(&mut wire, &stream).expect("write headers");
            for (index, (pts, keyframe)) in packets.iter().enumerate() {
                let packet = Packet {
                    pts: *pts,
                    dts: None,
                    keyframe: *keyframe,
                };
                muxer
                    .write_coded(&packet, &[index as u8; 4])
                    .expect("write coded packet");
            }
            muxer.finish().expect("finish");
        }

        let mut demuxer = Demuxer::open(&wire[..]).expect("read headers");
        assert_eq!(demuxer.stream().decode_delay, 2);
        assert_eq!(demuxer.stream().extradata, stream.extradata);
        assert_eq!(demuxer.stream().codec_name(), Some("h264"));

        let mut buf = Vec::new();
        let mut got = Vec::new();
        while let Some(packet) = demuxer.read_packet(&mut buf).expect("read coded packet") {
            got.push((packet, buf.clone()));
        }
        assert_eq!(got.len(), packets.len());
        for (index, ((packet, data), (pts, keyframe))) in got.iter().zip(&packets).enumerate() {
            assert_eq!(packet.pts, *pts, "packet {index} pts");
            assert_eq!(packet.keyframe, *keyframe, "packet {index} keyframe");
            assert_eq!(packet.dts, expected_dts[index], "packet {index} dts");
            assert_eq!(data, &vec![index as u8; 4], "packet {index} data");
        }
    }

    #[test]
    fn coded_aac_packets_round_trip_with_no_reordering() {
        let stream = a_coded_aac_stream();
        // Each packet is one AAC frame of 1024 samples; audio has no B-frames
        // so every packet is a keyframe and dts is pts.
        let packets: Vec<(i64, bool)> = (0..4i64).map(|i| (i * 1024, true)).collect();

        let mut wire = Vec::new();
        {
            let mut muxer = Muxer::new(&mut wire, &stream).expect("write headers");
            for (index, (pts, keyframe)) in packets.iter().enumerate() {
                let packet = Packet {
                    pts: *pts,
                    dts: None,
                    keyframe: *keyframe,
                };
                muxer
                    .write_coded(&packet, &[0x21, index as u8, 0x10, 0x04])
                    .expect("write coded packet");
            }
            muxer.finish().expect("finish");
        }

        let mut demuxer = Demuxer::open(&wire[..]).expect("read headers");
        assert_eq!(demuxer.stream().decode_delay, 0);
        assert_eq!(demuxer.stream().codec_name(), Some("aac"));

        let mut buf = Vec::new();
        let mut got = Vec::new();
        while let Some(packet) = demuxer.read_packet(&mut buf).expect("read coded packet") {
            got.push((packet, buf.clone()));
        }
        assert_eq!(got.len(), packets.len());
        for (index, ((packet, data), (pts, keyframe))) in got.iter().zip(&packets).enumerate() {
            assert_eq!(packet.pts, *pts, "packet {index} pts");
            assert_eq!(packet.dts, Some(*pts), "packet {index} dts: no reordering");
            assert_eq!(packet.keyframe, *keyframe, "packet {index} keyframe");
            assert_eq!(
                data,
                &vec![0x21, index as u8, 0x10, 0x04],
                "packet {index} data"
            );
        }
    }

    #[test]
    fn the_raw_path_writes_the_same_bytes_it_always_did() {
        // A pin over `write_frame`'s exact output, so a refactor that shares
        // machinery with `write_coded` cannot silently change it.
        let frames: Vec<(i64, Vec<u8>)> = (0..3i64).map(|i| (i, vec![i as u8; 16])).collect();
        let wire = wire_with(&frames);
        assert_eq!(
            Demuxer::open(&wire[..])
                .and_then(|mut d| {
                    let mut buf = Vec::new();
                    let mut out = Vec::new();
                    while let Some(pts) = d.read_frame(&mut buf)? {
                        out.push((pts, buf.clone()));
                    }
                    Ok(out)
                })
                .expect("round trip the raw wire"),
            frames
        );
    }
}
