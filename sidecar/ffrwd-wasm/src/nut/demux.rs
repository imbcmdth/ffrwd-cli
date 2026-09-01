//! Reading NUT: the headers that describe the stream, and the frames that
//! follow with the PTS they were coded with.

use std::collections::HashMap;
use std::io::Read;

use anyhow::{anyhow, bail, Result};

use super::bytes::{crc32, ByteReader};
use super::{
    flags, Media, Packet, Stream, TimeBase, ANNOTATION_CLASS, ANNOTATION_FOURCC,
    ANNOTATION_STREAM_ID, AUDIO_CLASS, FILE_ID, MAIN_STARTCODE, STREAM_STARTCODE,
    SYNCPOINT_STARTCODE, TRAILING_KEY, VERSION, VIDEO_CLASS,
};

const FILE_ID_LEN: usize = FILE_ID.len();

/// Largest packet body this demuxer will hold in memory to check. Headers and
/// syncpoints are tens of bytes; anything near this is damage.
const MAX_PARSED_BODY: u64 = 1 << 20;

/// Largest frame this demuxer will allocate for. Uncompressed 8K RGBA is a
/// tenth of it.
const MAX_FRAME_BYTES: u64 = 1 << 31;

/// Largest NDJSON payload one annotation packet may carry.
const MAX_ANNOTATION_BYTES: u64 = 1 << 20;

/// How many frames' rows may sit unclaimed before the annotation stream is
/// treated as unmatched to the video rather than merely ahead of it. Rows are
/// written just before their frame, so one is the working depth.
const MAX_UNCLAIMED_ROWS: usize = 1024;

/// A packet body wider than this carries its own header checksum ahead of the
/// data.
const HEADER_CHECKSUM_THRESHOLD: u64 = 4096;

/// The most elision headers a main header may declare, and the widest one.
/// ffmpeg's own bounds.
const MAX_ELISION_HEADERS: usize = 128;
const MAX_ELISION_BYTES: usize = 256;

/// The deepest frame reordering this wire carries. ffmpeg's own bound.
const MAX_DECODE_DELAY: u64 = 16;

/// One entry of the framecode table: what a single frame-header byte says
/// about the frame that follows it.
#[derive(Debug, Clone, Copy, Default)]
struct FrameCode {
    flags: u64,
    stream_id: u64,
    size_mul: u64,
    size_lsb: u64,
    pts_delta: i64,
    reserved_count: u64,
    header_idx: u64,
}

/// What the main header settles for the whole file.
struct MainHeader {
    stream_count: u64,
    time_bases: Vec<TimeBase>,
    frame_codes: [FrameCode; 256],
    /// The elision headers, entry 0 the empty one no frame names. A frame
    /// whose `header_idx` names another starts with that entry's bytes, and
    /// the wire carries only the rest.
    elision: Vec<Vec<u8>>,
}

/// One stream header, told apart by which stream it describes.
enum StreamHeader {
    Media(Stream),
    Annotations,
}

/// Reads one NUT stream: the subset described in this module's documentation.
pub struct Demuxer<R> {
    reader: ByteReader<R>,
    stream: Stream,
    annotations: bool,
    time_bases: Vec<TimeBase>,
    frame_codes: [FrameCode; 256],
    /// The elision headers the main header declared, entry 0 empty.
    elision: Vec<Vec<u8>>,
    /// Where each stream has got to, for a frame that codes only the low bits
    /// of its PTS. Indexed by stream id.
    last_pts: [i64; 2],
    /// The reorder buffer DTS falls out of: the `decode_delay + 1` most
    /// recent PTS, ascending, seeded with `None`. The smallest entry after a
    /// frame's PTS lands is that frame's DTS - `None` while a seed remains,
    /// which is the wire not settling it.
    pts_buffer: Vec<Option<i64>>,
    /// Rows read off the annotation stream, keyed by the PTS of the frame
    /// they belong to, until that frame arrives.
    unclaimed_rows: HashMap<i64, Vec<String>>,
    /// Rows the trailing record carried, which belong to no frame.
    trailing: Vec<String>,
}

impl<R: Read> Demuxer<R> {
    /// Reads up to and including the stream header, so geometry is known
    /// before the first frame arrives. One video stream and nothing else.
    pub fn open(inner: R) -> Result<Demuxer<R>> {
        Demuxer::read_headers(inner, false)
    }

    /// `open`, accepting the optional annotation stream beside the video.
    /// An input carrying only the video is still read.
    pub fn open_annotated(inner: R) -> Result<Demuxer<R>> {
        Demuxer::read_headers(inner, true)
    }

    fn read_headers(inner: R, annotations: bool) -> Result<Demuxer<R>> {
        let mut reader = ByteReader::new(inner);

        let mut id = [0u8; FILE_ID_LEN];
        reader.read_exact(&mut id, "the NUT identifier")?;
        if id != FILE_ID {
            bail!("input is not NUT: it does not begin with the NUT identifier");
        }

        let mut main: Option<MainHeader> = None;
        let mut stream: Option<Stream> = None;
        let mut annotation_stream = false;

        // Every stream header the main header declares has to be read before
        // the first frame, so the loop ends on the count rather than on the
        // video stream alone.
        while !headers_complete(main.as_ref(), stream.as_ref(), annotation_stream) {
            let first = reader
                .read_u8_or_eof()?
                .ok_or_else(|| anyhow!("NUT input ends before its stream header"))?;
            if first != b'N' {
                bail!("NUT input has a frame before its stream header");
            }
            let startcode = read_startcode(&mut reader, first)?;
            match startcode {
                MAIN_STARTCODE => {
                    if main.is_some() {
                        bail!("NUT input has a second main header");
                    }
                    let body = read_body(&mut reader, "the main header", true)?;
                    main = Some(parse_main(&body, annotations)?);
                }
                STREAM_STARTCODE => {
                    let Some(main) = main.as_ref() else {
                        bail!("NUT input has a stream header before its main header");
                    };
                    if !annotations && stream.is_some() {
                        bail!("NUT input carries more than one stream; this wire carries one");
                    }
                    let body = read_body(&mut reader, "a stream header", true)?;
                    match parse_stream(&body, main, annotations)? {
                        StreamHeader::Media(parsed) => {
                            if stream.is_some() {
                                bail!("NUT input carries more than one media stream; this wire carries one");
                            }
                            stream = Some(parsed);
                        }
                        StreamHeader::Annotations => {
                            if annotation_stream {
                                bail!("NUT input carries more than one annotation stream");
                            }
                            annotation_stream = true;
                        }
                    }
                }
                SYNCPOINT_STARTCODE => bail!("NUT input has a syncpoint before its stream header"),
                // Info packets, and anything this subset does not know, are
                // read past.
                _ => {
                    read_body(&mut reader, "a packet", false)?;
                }
            }
        }

        let main = main.expect("the loop runs until the main header is read");
        let stream = stream.expect("the loop runs until the stream header is read");
        let pts_buffer = vec![None; stream.decode_delay as usize + 1];
        Ok(Demuxer {
            reader,
            stream,
            annotations: annotation_stream,
            time_bases: main.time_bases,
            frame_codes: main.frame_codes,
            elision: main.elision,
            last_pts: [0; 2],
            pts_buffer,
            unclaimed_rows: HashMap::new(),
            trailing: Vec::new(),
        })
    }

    /// What the stream header said. An output header is built from this, so
    /// geometry and time base survive the hop.
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Whether the input actually carried an annotation stream. False for an
    /// input that declared only the video, even when annotations were asked
    /// for.
    pub fn has_annotations(&self) -> bool {
        self.annotations
    }

    /// The rows that arrived for the frame at `pts`, taken out of the
    /// demuxer. Empty when that frame carried none.
    pub fn take_rows(&mut self, pts: i64) -> Vec<String> {
        self.unclaimed_rows.remove(&pts).unwrap_or_default()
    }

    /// The rows the trailing record carried, taken out of the demuxer. The
    /// record comes after every frame, so this is empty until the stream has
    /// been read to its end.
    pub fn take_trailing(&mut self) -> Vec<String> {
        std::mem::take(&mut self.trailing)
    }

    /// The next frame's PTS, with its bytes in `data`, or None at the end of
    /// the stream. Packets between frames are consumed on the way.
    pub fn read_frame(&mut self, data: &mut Vec<u8>) -> Result<Option<i64>> {
        Ok(self.read_packet(data)?.map(|packet| packet.pts))
    }

    /// `read_frame`, keeping everything the frame's own header said: the PTS,
    /// the DTS the reorder buffer settles, and the keyframe flag. For an
    /// encoded stream this is the read that loses nothing.
    pub fn read_packet(&mut self, data: &mut Vec<u8>) -> Result<Option<Packet>> {
        loop {
            let Some(first) = self.reader.read_u8_or_eof()? else {
                return Ok(None);
            };

            if first == b'N' {
                let startcode = read_startcode(&mut self.reader, first)?;
                match startcode {
                    SYNCPOINT_STARTCODE => {
                        let body = read_body(&mut self.reader, "a syncpoint", true)?;
                        let pts = parse_syncpoint(&body, &self.time_bases, &self.stream)?;
                        self.last_pts = [pts; 2];
                    }
                    MAIN_STARTCODE | STREAM_STARTCODE => {
                        bail!("NUT input restates its headers mid-stream; this wire carries one stream with one set of headers")
                    }
                    // Info and index packets carry nothing a filter needs, and
                    // an unknown startcode is read past the same way.
                    _ => {
                        read_body(&mut self.reader, "a packet", false)?;
                    }
                }
                continue;
            }

            if let Some(keyframe) = self.read_frame_at(first, data)? {
                let pts = self.last_pts[0];
                return Ok(Some(Packet {
                    pts,
                    dts: self.decode_dts(pts),
                    keyframe,
                }));
            }
        }
    }

    /// One frame's DTS out of the reorder buffer: the new PTS lands over the
    /// smallest entry - the previous DTS, already spent - order is restored,
    /// and the smallest entry left is the answer. With no reordering the
    /// buffer holds one entry and the DTS is the PTS itself.
    fn decode_dts(&mut self, pts: i64) -> Option<i64> {
        self.pts_buffer[0] = Some(pts);
        let mut i = 0;
        while i + 1 < self.pts_buffer.len() && self.pts_buffer[i] > self.pts_buffer[i + 1] {
            self.pts_buffer.swap(i, i + 1);
            i += 1;
        }
        self.pts_buffer[0]
    }

    /// Reads the frame whose header byte is `first`. Returns its keyframe
    /// flag, or None for anything that is not a picture - an annotation
    /// packet, or a frame carrying none - which the caller reads past.
    fn read_frame_at(&mut self, first: u8, data: &mut Vec<u8>) -> Result<Option<bool>> {
        let code = self.frame_codes[usize::from(first)];
        if code.flags & flags::INVALID != 0 {
            bail!("NUT frame header byte {first} is not a framecode the main header defines");
        }

        self.reader.start_tap(&[first]);
        let mut frame_flags = code.flags;
        if frame_flags & flags::CODED != 0 {
            frame_flags ^= self.reader.read_v()?;
        }

        let mut stream_id = code.stream_id;
        if frame_flags & flags::STREAM_ID != 0 {
            stream_id = self.reader.read_v()?;
        }
        if stream_id != 0 && !(self.annotations && stream_id == ANNOTATION_STREAM_ID) {
            bail!("NUT frame belongs to stream {stream_id}; this wire carries stream 0 alone");
        }
        let slot = stream_id as usize;

        let pts = if frame_flags & flags::CODED_PTS != 0 {
            let coded = self.reader.read_v()?;
            self.decode_pts(slot, coded)?
        } else {
            self.last_pts[slot]
                .checked_add(code.pts_delta)
                .ok_or_else(|| anyhow!("NUT frame PTS overflows 64 bits"))?
        };

        let mut size = code.size_lsb;
        if frame_flags & flags::SIZE_MSB != 0 {
            let msb = self.reader.read_v()?;
            size = msb
                .checked_mul(code.size_mul)
                .and_then(|scaled| scaled.checked_add(code.size_lsb))
                .ok_or_else(|| anyhow!("NUT frame size overflows 64 bits"))?;
        }
        if frame_flags & flags::MATCH_TIME != 0 {
            self.reader.read_s()?;
        }
        let mut header_idx = code.header_idx;
        if frame_flags & flags::HEADER_IDX != 0 {
            header_idx = self.reader.read_v()?;
        }
        if header_idx as usize >= self.elision.len() {
            bail!(
                "NUT frame elides its first bytes into header {header_idx}, which the main header \
                 does not declare"
            );
        }
        let mut reserved_count = code.reserved_count;
        if frame_flags & flags::RESERVED != 0 {
            reserved_count = self.reader.read_v()?;
        }
        if reserved_count > 256 {
            bail!("NUT frame header declares {reserved_count} reserved fields");
        }
        for _ in 0..reserved_count {
            self.reader.read_v()?;
        }
        if frame_flags & flags::SM_DATA != 0 {
            bail!("NUT frame carries side data, which NUT version {VERSION} does not define");
        }

        let header = self.reader.take_tap();
        if frame_flags & flags::CHECKSUM != 0 {
            let want = self.reader.read_u32()?;
            let got = crc32(&header);
            if got != want {
                bail!("NUT frame header checksum is {got:#010x}, not the {want:#010x} it carries");
            }
        }

        self.last_pts[slot] = pts;

        // `size` counts the whole frame, elided prefix included; the wire
        // carries only what follows the prefix.
        let elided = self.elision[header_idx as usize].len();
        if elided as u64 > size {
            bail!(
                "NUT frame of {size} bytes elides a {elided} byte header, which does not fit in it"
            );
        }
        let on_wire = size - elided as u64;

        if frame_flags & flags::EOR != 0 {
            self.reader.skip(on_wire, "an end-of-relevance frame")?;
            return Ok(None);
        }

        if stream_id == ANNOTATION_STREAM_ID {
            if header_idx != 0 {
                bail!("NUT annotation packet elides its first bytes; that stream is written whole");
            }
            self.read_annotation(pts, size)?;
            return Ok(None);
        }

        if size > MAX_FRAME_BYTES {
            bail!("NUT frame claims {size} bytes, more than {MAX_FRAME_BYTES}");
        }
        data.clear();
        data.extend_from_slice(&self.elision[header_idx as usize]);
        data.resize(size as usize, 0);
        self.reader.read_exact(&mut data[elided..], "a frame")?;
        Ok(Some(frame_flags & flags::KEY != 0))
    }

    /// One annotation packet: the trailing record, or NDJSON with one row per
    /// line, held until the frame at `pts` asks for it.
    fn read_annotation(&mut self, pts: i64, size: u64) -> Result<()> {
        if size > MAX_ANNOTATION_BYTES {
            bail!("NUT annotation packet claims {size} bytes, more than {MAX_ANNOTATION_BYTES}");
        }
        if self.unclaimed_rows.len() >= MAX_UNCLAIMED_ROWS {
            bail!(
                "NUT annotation stream has {MAX_UNCLAIMED_ROWS} packets no frame claimed; its \
                 timestamps do not match the video's"
            );
        }
        let mut payload = vec![0u8; size as usize];
        self.reader
            .read_exact(&mut payload, "an annotation packet")?;
        let text = String::from_utf8(payload)
            .map_err(|_| anyhow!("NUT annotation packet at PTS {pts} is not UTF-8"))?;

        if let Some(rows) = trailing_rows(&text) {
            if !self.trailing.is_empty() {
                bail!("NUT annotation stream carries a second trailing record; there is one");
            }
            self.trailing = rows;
            return Ok(());
        }

        let rows: Vec<String> = text
            .lines()
            .filter(|line| !line.is_empty())
            .map(str::to_string)
            .collect();
        self.unclaimed_rows.entry(pts).or_default().extend(rows);
        Ok(())
    }

    /// A coded PTS is either the whole value, offset, or only its low bits,
    /// which are lifted back onto the last PTS seen on that stream.
    fn decode_pts(&self, slot: usize, coded: u64) -> Result<i64> {
        let shift = 1u64 << self.stream.msb_pts_shift;
        if coded >= shift {
            i64::try_from(coded - shift).map_err(|_| anyhow!("NUT frame PTS overflows 64 bits"))
        } else {
            let mask = (shift - 1) as i64;
            let delta = self.last_pts[slot] - mask / 2;
            Ok(((coded as i64 - delta) & mask) + delta)
        }
    }
}

/// The record trailing rows ride in. Its one key is what tells it from the
/// NDJSON a frame's rows are written as.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TrailingRecord<'a> {
    #[serde(borrow, rename = "trailing")]
    rows: Vec<&'a serde_json::value::RawValue>,
}

/// The rows a trailing record carries, verbatim, or None when `text` is a
/// frame's rows instead.
fn trailing_rows(text: &str) -> Option<Vec<String>> {
    debug_assert_eq!(TRAILING_KEY, "trailing", "the record's one key is renamed");
    let record: TrailingRecord = serde_json::from_str(text).ok()?;
    Some(
        record
            .rows
            .into_iter()
            .map(|row| row.get().to_string())
            .collect(),
    )
}

/// Whether every header the input declares has been read.
fn headers_complete(
    main: Option<&MainHeader>,
    stream: Option<&Stream>,
    annotation_stream: bool,
) -> bool {
    let Some(main) = main else {
        return false;
    };
    stream.is_some() && (main.stream_count < 2 || annotation_stream)
}

/// Reads the seven bytes of a startcode that follow the `N` already seen.
fn read_startcode<R: Read>(reader: &mut ByteReader<R>, first: u8) -> Result<u64> {
    let mut rest = [0u8; 7];
    reader.read_exact(&mut rest, "a startcode")?;
    let mut code = u64::from(first);
    for byte in rest {
        code = (code << 8) | u64::from(byte);
    }
    Ok(code)
}

/// Reads a packet's body. `parse` buffers it and checks its checksum; without
/// it the body is skipped, which is what info and index packets get.
fn read_body<R: Read>(reader: &mut ByteReader<R>, what: &str, parse: bool) -> Result<Vec<u8>> {
    let body_len = reader.read_v()?;
    // A packet this wide carries a checksum over its own header. Only info
    // and index packets ever reach that size here, and those are skipped
    // whole, so the field is read past rather than checked.
    if body_len > HEADER_CHECKSUM_THRESHOLD {
        reader.read_u32()?;
    }
    if body_len < 4 {
        bail!("NUT packet is {body_len} bytes, too short to hold its checksum ({what})");
    }
    if !parse {
        reader.skip(body_len, what)?;
        return Ok(Vec::new());
    }
    if body_len > MAX_PARSED_BODY {
        bail!("NUT packet claims {body_len} bytes, more than {MAX_PARSED_BODY} ({what})");
    }
    let mut body = vec![0u8; body_len as usize];
    reader.read_exact(&mut body, what)?;

    let split = body.len() - 4;
    let want = u32::from_be_bytes([
        body[split],
        body[split + 1],
        body[split + 2],
        body[split + 3],
    ]);
    body.truncate(split);
    let got = crc32(&body);
    if got != want {
        bail!("NUT packet checksum is {got:#010x}, not the {want:#010x} it carries ({what})");
    }
    Ok(body)
}

fn parse_main(body: &[u8], annotations: bool) -> Result<MainHeader> {
    let mut r = ByteReader::new(body);

    let version = r.read_v()?;
    if version != VERSION {
        bail!("NUT input is version {version}; this wire speaks version {VERSION}");
    }
    let stream_count = r.read_v()?;
    if annotations {
        if stream_count == 0 || stream_count > 2 {
            bail!("NUT input carries {stream_count} streams; this wire carries a video stream and an annotation stream");
        }
    } else if stream_count != 1 {
        bail!("NUT input carries {stream_count} streams; this wire carries one");
    }
    let _max_distance = r.read_v()?;

    let time_base_count = r.read_v()?;
    if time_base_count == 0 || time_base_count > 64 {
        bail!("NUT main header declares {time_base_count} time bases");
    }
    let mut time_bases = Vec::with_capacity(time_base_count as usize);
    for _ in 0..time_base_count {
        let num = r.read_v()?;
        let den = r.read_v()?;
        if num == 0 || den == 0 {
            bail!("NUT time base {num}/{den} is not a rate");
        }
        time_bases.push(TimeBase { num, den });
    }

    let frame_codes = parse_frame_codes(&mut r)?;

    // Entry 0 is the empty header no frame names; the ones declared here
    // follow it, so a frame's `header_idx` indexes this list directly.
    let header_count = r.read_v()?;
    if header_count as usize >= MAX_ELISION_HEADERS {
        bail!("NUT main header declares {header_count} elision headers");
    }
    let mut elision: Vec<Vec<u8>> = Vec::with_capacity(header_count as usize + 1);
    elision.push(Vec::new());
    for index in 0..header_count {
        let header = r.read_vb("an elision header")?;
        if header.is_empty() || header.len() > MAX_ELISION_BYTES {
            bail!("NUT elision header {} is {} bytes", index + 1, header.len());
        }
        elision.push(header);
    }

    Ok(MainHeader {
        stream_count,
        time_bases,
        frame_codes,
        elision,
    })
}

/// Walks the 256-entry framecode table. Each group in the header describes a
/// run of entries; index `N` is always invalid and never consumes one.
fn parse_frame_codes(r: &mut ByteReader<&[u8]>) -> Result<[FrameCode; 256]> {
    let mut codes = [FrameCode::default(); 256];
    let mut index = 0usize;
    let mut groups = 0usize;

    let mut pts_delta = 0i64;
    let mut size_mul = 1u64;
    let mut stream_id = 0u64;
    let mut header_idx = 0u64;

    while index < 256 {
        groups += 1;
        if groups > 256 {
            bail!("NUT main header describes its framecode table in more than 256 groups");
        }

        let code_flags = r.read_v()?;
        let fields = r.read_v()?;
        if fields > 0 {
            pts_delta = r.read_s()?;
        }
        if fields > 1 {
            size_mul = r.read_v()?;
        }
        if fields > 2 {
            stream_id = r.read_v()?;
        }
        let size_lsb = if fields > 3 { r.read_v()? } else { 0 };
        let reserved_count = if fields > 4 { r.read_v()? } else { 0 };
        let count = if fields > 5 {
            r.read_v()?
        } else {
            size_mul.saturating_sub(size_lsb)
        };
        if fields > 6 {
            r.read_s()?;
        }
        if fields > 7 {
            header_idx = r.read_v()?;
        }
        for _ in 8..fields {
            r.read_v()?;
        }

        let mut taken = 0u64;
        while taken < count && index < 256 {
            if index == usize::from(b'N') {
                codes[index].flags = flags::INVALID;
                index += 1;
                continue;
            }
            codes[index] = FrameCode {
                flags: code_flags,
                stream_id,
                size_mul,
                size_lsb: size_lsb.saturating_add(taken),
                pts_delta,
                reserved_count,
                header_idx,
            };
            index += 1;
            taken += 1;
        }
    }
    Ok(codes)
}

fn parse_stream(body: &[u8], main: &MainHeader, annotations: bool) -> Result<StreamHeader> {
    let mut r = ByteReader::new(body);

    let stream_id = r.read_v()?;
    if stream_id != 0 && !(annotations && stream_id == ANNOTATION_STREAM_ID) {
        bail!("NUT stream header is for stream {stream_id}; this wire carries stream 0 alone");
    }
    let stream_class = r.read_v()?;
    if stream_id == 0 && stream_class != VIDEO_CLASS && stream_class != AUDIO_CLASS {
        bail!(
            "NUT input carries stream class {stream_class}; this wire carries video (class \
             {VIDEO_CLASS}) and audio (class {AUDIO_CLASS})"
        );
    }
    let fourcc = r.read_vb("a codec tag")?;
    // The annotation stream stops here: a data class carries no geometry, and
    // nothing past the codec tag matters to a reader of NDJSON rows.
    if stream_id == ANNOTATION_STREAM_ID {
        if stream_class != ANNOTATION_CLASS || fourcc != ANNOTATION_FOURCC {
            bail!(
                "NUT stream 1 carries class {stream_class} codec {}; the annotation stream is \
                 class {ANNOTATION_CLASS} codec {}",
                String::from_utf8_lossy(&fourcc),
                String::from_utf8_lossy(ANNOTATION_FOURCC)
            );
        }
        return Ok(StreamHeader::Annotations);
    }
    let time_base_id = r.read_v()?;
    let time_base = *main
        .time_bases
        .get(usize::try_from(time_base_id).unwrap_or(usize::MAX))
        .ok_or_else(|| anyhow!("NUT stream header names time base {time_base_id}, which the main header does not declare"))?;
    let msb_pts_shift = r.read_v()?;
    if msb_pts_shift >= 63 {
        bail!("NUT stream header shifts PTS by {msb_pts_shift} bits");
    }
    let max_pts_distance = r.read_v()?;
    let decode_delay = r.read_v()?;
    if decode_delay > MAX_DECODE_DELAY {
        bail!(
            "NUT stream reorders frames by {decode_delay}, deeper than the {MAX_DECODE_DELAY} \
             this wire carries"
        );
    }
    let _stream_flags = r.read_v()?;
    let extradata = r.read_vb("codec specific data")?;

    let media = if stream_class == AUDIO_CLASS {
        parse_audio_geometry(&mut r)?
    } else {
        parse_video_geometry(&mut r)?
    };

    let stream = Stream {
        fourcc,
        time_base,
        msb_pts_shift: msb_pts_shift as u32,
        max_pts_distance,
        decode_delay,
        extradata,
        media,
    };
    if decode_delay != 0 && stream.codec_name().is_none() {
        bail!(
            "NUT stream of {} declares decode delay {decode_delay}; only an encoded \
             stream reorders frames",
            stream.fourcc_name()
        );
    }
    Ok(StreamHeader::Media(stream))
}

/// The fields a video stream header ends with.
fn parse_video_geometry(r: &mut ByteReader<&[u8]>) -> Result<Media> {
    let width = r.read_v()?;
    let height = r.read_v()?;
    if width == 0 || height == 0 {
        bail!("NUT stream header gives frame size {width}x{height}");
    }
    Ok(Media::Video {
        width: u32::try_from(width).map_err(|_| anyhow!("NUT frame width {width} is too large"))?,
        height: u32::try_from(height)
            .map_err(|_| anyhow!("NUT frame height {height} is too large"))?,
        sample_width: r.read_v()?,
        sample_height: r.read_v()?,
        colorspace_type: r.read_v()?,
    })
}

/// The fields an audio stream header ends with. The rate is a ratio in NUT;
/// this wire carries whole rates, so a denominator that does not divide it is
/// refused rather than rounded.
fn parse_audio_geometry(r: &mut ByteReader<&[u8]>) -> Result<Media> {
    let num = r.read_v()?;
    let den = r.read_v()?;
    let channels = r.read_v()?;
    if den == 0 || num == 0 || !num.is_multiple_of(den) {
        bail!("NUT audio stream declares sample rate {num}/{den}; this wire carries whole rates");
    }
    if channels == 0 {
        bail!("NUT audio stream declares {channels} channels");
    }
    Ok(Media::Audio {
        sample_rate: u32::try_from(num / den)
            .map_err(|_| anyhow!("NUT sample rate {num}/{den} is too large"))?,
        channels: u32::try_from(channels)
            .map_err(|_| anyhow!("NUT channel count {channels} is too large"))?,
    })
}

/// A syncpoint restates where the stream is, which is what a frame coding
/// only the low bits of its PTS is measured from.
fn parse_syncpoint(body: &[u8], time_bases: &[TimeBase], stream: &Stream) -> Result<i64> {
    let mut r = ByteReader::new(body);
    let coded = r.read_v()?;
    let count = time_bases.len() as u64;
    let index = (coded % count) as usize;
    let ts = (coded / count) as i128;
    let from = time_bases[index];
    let to = stream.time_base;
    let scaled = ts * i128::from(from.num) * i128::from(to.den)
        / (i128::from(from.den) * i128::from(to.num));
    i64::try_from(scaled).map_err(|_| anyhow!("NUT syncpoint timestamp overflows 64 bits"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nut::mux::Muxer;
    use crate::nut::{fourcc_for_pix_fmt, Stream, TimeBase};

    fn a_stream() -> Stream {
        Stream::video("rgba", 2, 2, TimeBase { num: 1, den: 25 }).expect("rgba is carried")
    }

    /// The message `open` refuses `wire` with.
    fn refusal(wire: &[u8]) -> String {
        match Demuxer::open(wire) {
            Ok(_) => panic!("this stream should have been refused"),
            Err(e) => e.to_string(),
        }
    }

    /// One NUT stream with a single 16-byte frame.
    fn one_frame_wire() -> Vec<u8> {
        let mut wire = Vec::new();
        {
            let mut muxer = Muxer::new(&mut wire, &a_stream()).expect("write headers");
            muxer.write_frame(0, &[0u8; 16]).expect("write frame");
            muxer.finish().expect("finish");
        }
        wire
    }

    #[test]
    fn a_stream_that_is_not_nut_is_named() {
        let err = refusal(b"rawvideo bytes, no header at all");
        assert!(err.contains("not NUT"), "{err}");
    }

    #[test]
    fn a_truncated_frame_is_an_error() {
        let wire = one_frame_wire();
        let mut demuxer = Demuxer::open(&wire[..wire.len() - 4]).expect("headers are intact");
        let err = demuxer.read_frame(&mut Vec::new()).unwrap_err().to_string();
        assert!(err.contains("ends inside a frame"), "{err}");
    }

    #[test]
    fn a_corrupt_header_checksum_is_an_error() {
        let mut wire = one_frame_wire();
        // The main header's body starts right after the identifier, the
        // startcode and a one-byte forward pointer; flip a bit in it.
        wire[FILE_ID_LEN + 9] ^= 0x01;
        let err = refusal(&wire);
        assert!(err.contains("checksum"), "{err}");
    }

    #[test]
    fn a_main_header_naming_two_streams_is_refused() {
        let mut wire = one_frame_wire();
        // The main header's length follows the identifier and the startcode,
        // and its body starts after that: version, then the stream count.
        let length_at = FILE_ID_LEN + 8;
        assert!(wire[length_at] < 0x80, "the main header body is small");
        let len = usize::from(wire[length_at]);
        let body = length_at + 1;
        wire[body + 1] = 2;
        let checksum = crc32(&wire[body..body + len - 4]).to_be_bytes();
        wire[body + len - 4..body + len].copy_from_slice(&checksum);

        let err = refusal(&wire);
        assert!(err.contains("2 streams"), "{err}");
    }

    #[test]
    fn a_codec_tag_survives_the_headers() {
        let stream = Stream::video("yuv420p", 4, 4, TimeBase { num: 1, den: 30 })
            .expect("yuv420p is carried");
        let mut wire = Vec::new();
        {
            let mut muxer = Muxer::new(&mut wire, &stream).expect("write headers");
            muxer.write_frame(3, &[7u8; 24]).expect("write frame");
            muxer.finish().expect("finish");
        }

        let demuxer = Demuxer::open(&wire[..]).expect("read headers");
        assert_eq!(
            demuxer.stream().fourcc,
            fourcc_for_pix_fmt("yuv420p").unwrap()
        );
        assert_eq!(demuxer.stream().pix_fmt(), Some("yuv420p"));
        assert_eq!(demuxer.stream().video_geometry(), Some((4, 4)));
        assert_eq!(demuxer.stream().time_base, TimeBase { num: 1, den: 30 });
    }
}
