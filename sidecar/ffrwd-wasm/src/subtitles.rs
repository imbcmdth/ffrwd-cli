//! Cue rows written as subtitles: a whole document, or a stream of cues.
//!
//! NUT carries no per-packet duration, so a cue muxed into the edge stream
//! would lose its end time. A document (SRT, WebVTT) gathers the cues and is
//! written whole when the stream ends, an ordinary subtitle file the next
//! ffmpeg reads as an input. ffmpeg reads such a file to its end before it
//! opens the inputs after it, so a stream that never ends needs the other
//! form: WebVTT in WebM, each cue written the moment its row arrives with its
//! end time in the block, which ffmpeg reads as it goes.
//!
//! ffmpeg reads no input more than a tenth of a second ahead of an output's
//! furthest-behind stream, and a cue stream falls behind between
//! utterances, so a stream also carries heartbeats: a cue lasting no time
//! at the time the host has reached, written as that time moves on. ffmpeg
//! counts them, and the compiler drops them before the muxer
//! (`noise=drop=not(duration)`), so no file holds one.
//!
//! Only a finished cue is written. A row whose `final` is false is an
//! utterance still growing, sent again as it grows, and would make cues that
//! overlap; a row with no `final` counts as finished. A row's `voice` names
//! who speaks, and WebVTT marks the cue with it as `<v Name>`.
//!
//! A subtitle document holds TEXT. A row whose payload is a vector belongs
//! in a rows file or inside the stream's own packets, and the compiler
//! refuses the track before a run gets here.

use anyhow::{bail, Result};
use serde_json::{Map, Value};

/// What a heartbeat says: a space, since ffmpeg's matroska demuxer refuses a
/// WebVTT block with no text.
const HEARTBEAT: &str = " ";

/// The names a cue row carries: two bounds and its text, and whether it is
/// finished and who says it where the producer knows.
const START: &str = "start_t";
const END: &str = "end_t";
const TEXT: &str = "text";
const FINAL: &str = "final";
const VOICE: &str = "voice";

/// What a subtitle output writes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format {
    Srt,
    WebVtt,
    /// WebVTT cues in WebM, one block a cue, written as they arrive.
    WebM,
}

impl Format {
    /// The `-f` value that asks for this format.
    pub fn name(self) -> &'static str {
        match self {
            Format::Srt => "srt",
            Format::WebVtt => "webvtt",
            Format::WebM => "webm",
        }
    }

    /// What separates a timestamp's seconds from its milliseconds.
    fn decimal(self) -> char {
        match self {
            Format::Srt => ',',
            Format::WebVtt | Format::WebM => '.',
        }
    }
}

/// One cue: when it shows, when it goes, what it says and who says it.
#[derive(Debug, PartialEq)]
pub struct Cue {
    start: f64,
    end: f64,
    text: String,
    voice: Option<String>,
}

/// A time in seconds as a subtitle timestamp, `HH:MM:SS` and milliseconds.
fn timestamp(seconds: f64, decimal: char) -> String {
    let total = millis(seconds);
    let (millis, seconds) = (total % 1000, total / 1000);
    let (secs, minutes) = (seconds % 60, seconds / 60);
    let (mins, hours) = (minutes % 60, minutes / 60);
    format!("{hours:02}:{mins:02}:{secs:02}{decimal}{millis:03}")
}

/// A time in seconds as whole milliseconds, never below zero.
fn millis(seconds: f64) -> u64 {
    (seconds * 1000.0).round().max(0.0) as u64
}

/// One of a cue row's times, refused naming the output and the field when it
/// is missing or is not a time.
fn time(object: &Map<String, Value>, field: &str, output: &str, row: &str) -> Result<f64> {
    let Some(value) = object.get(field) else {
        bail!("{output}: a cue row carries {START}, {END} and {TEXT}, and this one has no {field}: {row}");
    };
    let Some(seconds) = value.as_f64() else {
        bail!("{output}: a cue row's {field} is a number of seconds, and this one is not: {row}");
    };
    if !seconds.is_finite() || seconds < 0.0 {
        bail!(
            "{output}: a cue row's {field} is a time in seconds, and {seconds} is not one: {row}"
        );
    }
    Ok(seconds)
}

/// The cue one row carries, or None when the row is not a cue at all or is
/// one still growing. A row object naming none of the three is the other arm
/// of a module's rows schema, a trailing summary record say, and is passed
/// over rather than refused. One naming some of them is a cue that does not
/// hold together, and names what it is missing.
pub fn read_row(row: &str, output: &str) -> Result<Option<Cue>> {
    let value: Value = match serde_json::from_str(row) {
        Ok(value) => value,
        Err(e) => {
            bail!("{output}: a subtitle output reads cue rows, and this row is not JSON: {e}")
        }
    };
    let Some(object) = value.as_object() else {
        bail!(
            "{output}: a subtitle output reads cue rows, and this row is not a JSON object: {row}"
        );
    };
    if [START, END, TEXT].iter().all(|f| !object.contains_key(*f)) {
        return Ok(None);
    }
    match object.get(FINAL) {
        None | Some(Value::Null) | Some(Value::Bool(true)) => {}
        Some(Value::Bool(false)) => return Ok(None),
        Some(_) => {
            bail!("{output}: a cue row's {FINAL} is true or false, and this one is neither: {row}")
        }
    }
    let voice = match object.get(VOICE) {
        None | Some(Value::Null) => None,
        Some(Value::String(name)) if name.trim().is_empty() => None,
        Some(Value::String(name)) => Some(name.trim().to_string()),
        Some(_) => bail!("{output}: a cue row's {VOICE} is a name, and this one is not: {row}"),
    };

    let start = time(object, START, output, row)?;
    let end = time(object, END, output, row)?;
    let Some(value) = object.get(TEXT) else {
        bail!("{output}: a cue row carries {START}, {END} and {TEXT}, and this one has no {TEXT}: {row}");
    };
    let Some(text) = value.as_str() else {
        bail!("{output}: a cue row's {TEXT} is a string, and this one is not: {row}");
    };
    if text.trim().is_empty() {
        bail!("{output}: a cue row's {TEXT} is what it says, and this one says nothing: {row}");
    }
    let text = text.to_string();
    if end < start {
        bail!("{output}: a cue row ends at {end} and starts at {start}, so it ends before it starts: {row}");
    }

    Ok(Some(Cue {
        start,
        end,
        text,
        voice,
    }))
}

/// A cue's text as WebVTT holds it: `&`, `<` and `>` as character
/// references, no blank line, which would end the cue, and no `-->`, which
/// would start one. A voice marks the whole cue, `<v Name>`.
fn webvtt_text(cue: &Cue) -> String {
    let escaped = escape(&cue.text);
    let lines: Vec<&str> = escaped
        .lines()
        .map(str::trim_end)
        .filter(|line| !line.trim().is_empty())
        .collect();
    let text = lines.join("\n");
    match &cue.voice {
        Some(name) => {
            let name = escape(&name.split_whitespace().collect::<Vec<_>>().join(" "));
            format!("<v {name}>{text}")
        }
        None => text,
    }
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// The cues one output has been handed, and what each row adds to it.
pub struct Writer {
    format: Format,
    /// A document's cues so far, in the order their rows arrived.
    cues: Vec<Cue>,
    /// The start of the last cue a stream wrote, in milliseconds.
    last_start: u64,
    /// The time of the last heartbeat a stream wrote, in milliseconds.
    last_beat: Option<u64>,
}

impl Writer {
    pub fn new(format: Format) -> Writer {
        Writer {
            format,
            cues: Vec::new(),
            last_start: 0,
            last_beat: None,
        }
    }

    /// What a stream writes once the host's time has reached `seconds`: a
    /// heartbeat, a cue lasting no time, when that time is past the last one.
    /// A cue still to come keeps its own start, however far back that is,
    /// since a heartbeat orders nothing: it never reaches a file. A document
    /// writes nothing until it ends.
    pub fn progress(&mut self, seconds: f64) -> Vec<u8> {
        if self.format != Format::WebM {
            return Vec::new();
        }
        let at = millis(seconds);
        if self.last_beat.is_some_and(|last| at <= last) {
            return Vec::new();
        }
        self.last_beat = Some(at);
        webm::cue(at, 0, HEARTBEAT)
    }

    /// What the output starts with, written before any row arrives: a
    /// stream's header, so its reader can open it while the cues are still to
    /// come. A document starts with nothing; it is written whole in `finish`.
    pub fn open(&self) -> Vec<u8> {
        match self.format {
            Format::WebM => webm::header(),
            Format::Srt | Format::WebVtt => Vec::new(),
        }
    }

    /// Reads one row: a document keeps the cue it carries, and a stream hands
    /// back the cue's block to write now. A row that is not a cue, or not a
    /// finished one, adds nothing.
    ///
    /// A stream's cues go out in the order they start, since its reader takes
    /// each as the next packet. A cue whose row arrives after a later one's
    /// starts where that one did, and one that ended by then is dropped.
    pub fn push_row(&mut self, row: &str, output: &str) -> Result<Vec<u8>> {
        let Some(cue) = read_row(row, output)? else {
            return Ok(Vec::new());
        };
        if self.format != Format::WebM {
            self.cues.push(cue);
            return Ok(Vec::new());
        }
        let start = millis(cue.start).max(self.last_start);
        let end = millis(cue.end);
        if end <= start {
            return Ok(Vec::new());
        }
        self.last_start = start;
        Ok(webm::cue(start, end - start, &webvtt_text(&cue)))
    }

    /// What is left to write once the stream has ended: the whole document,
    /// numbered from one in the order the cues arrived. A stream has written
    /// everything already.
    pub fn finish(&mut self) -> Vec<u8> {
        if self.format == Format::WebM {
            return Vec::new();
        }
        let decimal = self.format.decimal();
        let mut out = String::new();
        if self.format == Format::WebVtt {
            out.push_str("WEBVTT\n\n");
        }
        for (index, cue) in self.cues.iter().enumerate() {
            out.push_str(&format!("{}\n", index + 1));
            out.push_str(&format!(
                "{} --> {}\n",
                timestamp(cue.start, decimal),
                timestamp(cue.end, decimal)
            ));
            if self.format == Format::WebVtt {
                out.push_str(&webvtt_text(cue));
            } else {
                out.push_str(&cue.text);
            }
            out.push_str("\n\n");
        }
        out.into_bytes()
    }
}

/// WebVTT in WebM, as ffmpeg's matroska demuxer reads it from a pipe: an EBML
/// header, a segment of unknown size whose one track is `D_WEBVTT/SUBTITLES`,
/// an empty cluster, and then a cluster per cue. The demuxer reads its header
/// up to the first cluster, so the empty one is what lets ffmpeg open the
/// input, and go on to read its others, before any cue has been said. A
/// block's data is the cue's identifier line, its settings line, both empty
/// here, and its text; its duration is the cue's. Times are milliseconds.
mod webm {
    const EBML: u32 = 0x1A45_DFA3;
    const EBML_VERSION: u32 = 0x4286;
    const EBML_READ_VERSION: u32 = 0x42F7;
    const EBML_MAX_ID_LENGTH: u32 = 0x42F2;
    const EBML_MAX_SIZE_LENGTH: u32 = 0x42F3;
    const DOC_TYPE: u32 = 0x4282;
    const DOC_TYPE_VERSION: u32 = 0x4287;
    const DOC_TYPE_READ_VERSION: u32 = 0x4285;
    const SEGMENT: u32 = 0x1853_8067;
    const INFO: u32 = 0x1549_A966;
    const TIMESTAMP_SCALE: u32 = 0x2A_D7B1;
    const MUXING_APP: u32 = 0x4D80;
    const WRITING_APP: u32 = 0x5741;
    const TRACKS: u32 = 0x1654_AE6B;
    const TRACK_ENTRY: u32 = 0xAE;
    const TRACK_NUMBER: u32 = 0xD7;
    const TRACK_UID: u32 = 0x73C5;
    const TRACK_TYPE: u32 = 0x83;
    const FLAG_LACING: u32 = 0x9C;
    const CODEC_ID: u32 = 0x86;
    const LANGUAGE: u32 = 0x22_B59C;
    const CLUSTER: u32 = 0x1F43_B675;
    const TIMESTAMP: u32 = 0xE7;
    const BLOCK_GROUP: u32 = 0xA0;
    const BLOCK: u32 = 0xA1;
    const BLOCK_DURATION: u32 = 0x9B;

    /// A segment's size when it is not known: every bit of an eight-byte
    /// size set.
    const UNKNOWN_SIZE: [u8; 8] = [0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
    /// Matroska's track type for subtitles.
    const SUBTITLE_TRACK: u64 = 0x11;

    fn id(id: u32) -> Vec<u8> {
        let bytes = id.to_be_bytes();
        let first = bytes.iter().position(|b| *b != 0).unwrap_or(3);
        bytes[first..].to_vec()
    }

    /// An element's size as EBML's variable-length integer, in the fewest
    /// bytes that hold it. All ones is reserved for a size not known.
    fn size(n: usize) -> Vec<u8> {
        let n = n as u64;
        let mut length = 1;
        while length < 8 && n >= (1u64 << (7 * length)) - 1 {
            length += 1;
        }
        let marked = n | (1u64 << (7 * length));
        marked.to_be_bytes()[8 - length..].to_vec()
    }

    fn element(element: u32, payload: &[u8]) -> Vec<u8> {
        let mut out = id(element);
        out.extend(size(payload.len()));
        out.extend_from_slice(payload);
        out
    }

    fn uint(element_id: u32, value: u64) -> Vec<u8> {
        let bytes = value.to_be_bytes();
        let first = bytes.iter().position(|b| *b != 0).unwrap_or(7);
        element(element_id, &bytes[first..])
    }

    fn text(element_id: u32, value: &str) -> Vec<u8> {
        element(element_id, value.as_bytes())
    }

    pub fn header() -> Vec<u8> {
        let mut out = element(
            EBML,
            &[
                uint(EBML_VERSION, 1),
                uint(EBML_READ_VERSION, 1),
                uint(EBML_MAX_ID_LENGTH, 4),
                uint(EBML_MAX_SIZE_LENGTH, 8),
                text(DOC_TYPE, "webm"),
                uint(DOC_TYPE_VERSION, 4),
                uint(DOC_TYPE_READ_VERSION, 2),
            ]
            .concat(),
        );
        out.extend(id(SEGMENT));
        out.extend(UNKNOWN_SIZE);
        let app = concat!("ffrwd-wasm ", env!("CARGO_PKG_VERSION"));
        out.extend(element(
            INFO,
            &[
                uint(TIMESTAMP_SCALE, 1_000_000),
                text(MUXING_APP, app),
                text(WRITING_APP, app),
            ]
            .concat(),
        ));
        let track = [
            uint(TRACK_NUMBER, 1),
            uint(TRACK_UID, 1),
            uint(TRACK_TYPE, SUBTITLE_TRACK),
            uint(FLAG_LACING, 0),
            text(CODEC_ID, "D_WEBVTT/SUBTITLES"),
            // Matroska reads a track that names no language as English.
            text(LANGUAGE, "und"),
        ]
        .concat();
        out.extend(element(TRACKS, &element(TRACK_ENTRY, &track)));
        out.extend(element(CLUSTER, &uint(TIMESTAMP, 0)));
        out
    }

    /// One cue as a cluster of its own, so its time is the cluster's and the
    /// block's own offset is zero.
    pub fn cue(start: u64, duration: u64, text: &str) -> Vec<u8> {
        let mut block = vec![0x81, 0x00, 0x00, 0x00];
        block.extend_from_slice(b"\n\n");
        block.extend_from_slice(text.as_bytes());
        let group = [element(BLOCK, &block), uint(BLOCK_DURATION, duration)].concat();
        element(
            CLUSTER,
            &[uint(TIMESTAMP, start), element(BLOCK_GROUP, &group)].concat(),
        )
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn a_size_takes_the_fewest_bytes_that_hold_it() {
            assert_eq!(size(0), vec![0x80]);
            assert_eq!(size(126), vec![0xFE]);
            // 127 in one byte is all ones, which reads as a size not known.
            assert_eq!(size(127), vec![0x40, 0x7F]);
            assert_eq!(size(300), vec![0x41, 0x2C]);
        }

        #[test]
        fn an_id_keeps_its_own_length() {
            assert_eq!(id(TRACK_ENTRY), vec![0xAE]);
            assert_eq!(id(TRACK_UID), vec![0x73, 0xC5]);
            assert_eq!(id(SEGMENT), vec![0x18, 0x53, 0x80, 0x67]);
        }

        #[test]
        fn a_cue_is_a_cluster_at_its_start_holding_its_text_and_duration() {
            let body = [
                vec![0xE7, 0x82, 0x05, 0xDC],
                vec![0xA0, 0x90],
                vec![0xA1, 0x8A, 0x81, 0x00, 0x00, 0x00, b'\n', b'\n'],
                b"hola".to_vec(),
                vec![0x9B, 0x82, 0x07, 0xD0],
            ]
            .concat();
            let mut expected = vec![0x1F, 0x43, 0xB6, 0x75];
            expected.extend(size(body.len()));
            expected.extend(body);
            assert_eq!(cue(1500, 2000, "hola"), expected);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OUT: &str = "-f srt subs.srt";

    fn cue(row: &str) -> Cue {
        read_row(row, OUT)
            .unwrap_or_else(|e| panic!("reading {row}: {e}"))
            .unwrap_or_else(|| panic!("{row} is a cue"))
    }

    fn written(format: Format, rows: &[&str]) -> String {
        let mut writer = Writer::new(format);
        for row in rows {
            writer.push_row(row, OUT).expect("a row");
        }
        String::from_utf8(writer.finish()).expect("text")
    }

    #[test]
    fn a_row_with_all_three_names_is_a_cue() {
        assert_eq!(
            cue(r#"{"start_t":1.5,"end_t":3.25,"text":"hola"}"#),
            Cue {
                start: 1.5,
                end: 3.25,
                text: "hola".to_string(),
                voice: None,
            }
        );
    }

    #[test]
    fn a_row_naming_none_of_them_is_passed_over() {
        // The other arm of a module's rows schema: transcribe's trailing
        // record, which rides the same output.
        assert_eq!(
            read_row(r#"{"transcript":"hola que tal"}"#, OUT).expect("not a refusal"),
            None
        );
        assert_eq!(read_row("{}", OUT).expect("not a refusal"), None);
    }

    #[test]
    fn only_a_finished_cue_is_one() {
        let growing = r#"{"start_t":1.0,"end_t":1.4,"text":"ho","final":false}"#;
        assert_eq!(read_row(growing, OUT).expect("not a refusal"), None);
        let done = cue(r#"{"start_t":1.0,"end_t":2.0,"text":"hola","final":true}"#);
        assert_eq!(done.text, "hola");
        let err = read_row(
            r#"{"start_t":1.0,"end_t":2.0,"text":"hola","final":"yes"}"#,
            OUT,
        )
        .expect_err("refused");
        assert!(err.to_string().contains("final"), "got: {err}");
    }

    #[test]
    fn a_voice_is_who_says_the_cue() {
        let said = cue(r#"{"start_t":1.0,"end_t":2.0,"text":"hola","voice":" Ana "}"#);
        assert_eq!(said.voice.as_deref(), Some("Ana"));
        let unsaid = cue(r#"{"start_t":1.0,"end_t":2.0,"text":"hola","voice":null}"#);
        assert_eq!(unsaid.voice, None);
        let err = read_row(
            r#"{"start_t":1.0,"end_t":2.0,"text":"hola","voice":3}"#,
            OUT,
        )
        .expect_err("refused");
        assert!(err.to_string().contains("voice"), "got: {err}");
    }

    #[test]
    fn a_row_missing_one_of_them_is_refused_naming_the_output_and_the_field() {
        let err = read_row(r#"{"start_t":1.0,"text":"hola"}"#, OUT).expect_err("refused");
        let message = err.to_string();
        assert!(message.contains(OUT), "the output is named, got: {message}");
        assert!(
            message.contains("end_t"),
            "the field is named, got: {message}"
        );
    }

    #[test]
    fn a_time_that_is_not_a_time_is_refused_naming_the_field() {
        let err =
            read_row(r#"{"start_t":"soon","end_t":2.0,"text":"hola"}"#, OUT).expect_err("refused");
        assert!(err.to_string().contains("start_t"), "got: {err}");

        let err =
            read_row(r#"{"start_t":-1.0,"end_t":2.0,"text":"hola"}"#, OUT).expect_err("refused");
        assert!(err.to_string().contains("start_t"), "got: {err}");
    }

    #[test]
    fn a_cue_that_ends_before_it_starts_is_refused() {
        let err =
            read_row(r#"{"start_t":3.0,"end_t":1.0,"text":"hola"}"#, OUT).expect_err("refused");
        assert!(err.to_string().contains(OUT), "got: {err}");
    }

    #[test]
    fn a_row_that_is_not_an_object_is_refused_naming_the_output() {
        for row in [r#"["start_t"]"#, "42", "not json at all"] {
            let err = read_row(row, OUT).expect_err("refused");
            assert!(err.to_string().contains(OUT), "{row} gave: {err}");
        }
    }

    #[test]
    fn a_timestamp_is_hours_minutes_seconds_and_milliseconds() {
        assert_eq!(timestamp(0.0, ','), "00:00:00,000");
        assert_eq!(timestamp(1.5, ','), "00:00:01,500");
        assert_eq!(timestamp(3661.25, ','), "01:01:01,250");
        assert_eq!(timestamp(1.5, '.'), "00:00:01.500");
    }

    #[test]
    fn srt_numbers_its_cues_from_one_in_the_order_they_arrived() {
        // Two cues, and neither the record that is not a cue nor the one
        // still growing is one of them. SRT has no voices.
        assert_eq!(
            written(
                Format::Srt,
                &[
                    r#"{"start_t":0.0,"end_t":2.0,"text":"uno"}"#,
                    r#"{"transcript":"uno dos"}"#,
                    r#"{"start_t":2.5,"end_t":3.0,"text":"do","final":false}"#,
                    r#"{"start_t":2.5,"end_t":4.0,"text":"dos","voice":"Ana"}"#,
                ]
            ),
            "1\n00:00:00,000 --> 00:00:02,000\nuno\n\n\
             2\n00:00:02,500 --> 00:00:04,000\ndos\n\n"
        );
    }

    #[test]
    fn webvtt_opens_with_its_header_and_marks_voices_and_markup() {
        assert_eq!(
            written(
                Format::WebVtt,
                &[
                    r#"{"start_t":0.0,"end_t":2.0,"text":"uno"}"#,
                    r#"{"start_t":2.0,"end_t":3.0,"text":"a < b & c\n\nd --> e","voice":"Ana <1>"}"#,
                ]
            ),
            "WEBVTT\n\n1\n00:00:00.000 --> 00:00:02.000\nuno\n\n\
             2\n00:00:02.000 --> 00:00:03.000\n<v Ana &lt;1&gt;>a &lt; b &amp; c\nd --&gt; e\n\n"
        );
    }

    #[test]
    fn a_webm_stream_writes_each_finished_cue_as_it_arrives() {
        let mut writer = Writer::new(Format::WebM);
        let opening = writer.open();
        assert!(opening.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]));
        // An empty cluster closes the header, so a reader opens it at once.
        assert!(opening.ends_with(&[0x1F, 0x43, 0xB6, 0x75, 0x83, 0xE7, 0x81, 0x00]));
        assert!(writer
            .push_row(
                r#"{"start_t":1.0,"end_t":1.5,"text":"ho","final":false}"#,
                OUT
            )
            .expect("a row")
            .is_empty());
        assert_eq!(
            writer
                .push_row(
                    r#"{"start_t":1.0,"end_t":2.5,"text":"hola","voice":"Ana","final":true}"#,
                    OUT
                )
                .expect("a row"),
            webm::cue(1000, 1500, "<v Ana>hola")
        );
        // A cue arriving after a later one starts where that one did...
        assert_eq!(
            writer
                .push_row(r#"{"start_t":0.5,"end_t":3.0,"text":"tarde"}"#, OUT)
                .expect("a row"),
            webm::cue(1000, 2000, "tarde")
        );
        // ...and one that ended by then is gone.
        assert!(writer
            .push_row(r#"{"start_t":0.2,"end_t":0.9,"text":"pasado"}"#, OUT)
            .expect("a row")
            .is_empty());
        assert!(writer.finish().is_empty());
    }

    #[test]
    fn a_stream_beats_as_time_moves_on_and_a_late_cue_keeps_its_start() {
        let mut writer = Writer::new(Format::WebM);
        assert_eq!(writer.progress(1.0), webm::cue(1000, 0, " "));
        // Time standing still beats no more.
        assert!(writer.progress(1.0).is_empty());
        assert_eq!(writer.progress(9.5), webm::cue(9500, 0, " "));
        // A cue said at 5 s, finished only now, is written at 5 s.
        assert_eq!(
            writer
                .push_row(r#"{"start_t":5.0,"end_t":6.0,"text":"tarde"}"#, OUT)
                .expect("a row"),
            webm::cue(5000, 1000, "tarde")
        );
        // A document has nothing to beat.
        assert!(Writer::new(Format::WebVtt).progress(1.0).is_empty());
    }
}
