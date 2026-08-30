//! Cue rows gathered into a subtitle document.
//!
//! NUT carries no per-packet duration, so a cue muxed into the edge stream
//! would lose its end time. The cues are all in hand by the end of the stream,
//! so they are gathered instead and written whole to an output of their own -
//! an ordinary subtitle file the next ffmpeg reads as an input.

use anyhow::{bail, Result};
use serde_json::{Map, Value};

/// The names a cue row carries, all three or none.
const START: &str = "start_t";
const END: &str = "end_t";
const TEXT: &str = "text";

/// Which document a subtitle output writes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format {
    Srt,
    WebVtt,
}

impl Format {
    /// The `-f` value that asks for this format.
    pub fn name(self) -> &'static str {
        match self {
            Format::Srt => "srt",
            Format::WebVtt => "webvtt",
        }
    }

    /// What separates a timestamp's seconds from its milliseconds.
    fn decimal(self) -> char {
        match self {
            Format::Srt => ',',
            Format::WebVtt => '.',
        }
    }
}

/// One cue: when it shows, when it goes, and what it says.
#[derive(Debug, PartialEq)]
pub struct Cue {
    start: f64,
    end: f64,
    text: String,
}

/// A time in seconds as a subtitle timestamp, `HH:MM:SS` and milliseconds.
fn timestamp(seconds: f64, decimal: char) -> String {
    let total = (seconds * 1000.0).round().max(0.0) as u64;
    let (millis, seconds) = (total % 1000, total / 1000);
    let (secs, minutes) = (seconds % 60, seconds / 60);
    let (mins, hours) = (minutes % 60, minutes / 60);
    format!("{hours:02}:{mins:02}:{secs:02}{decimal}{millis:03}")
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

/// The cue one row carries, or None when the row is not a cue at all. A row
/// object naming none of the three is the other arm of a module's rows schema,
/// a trailing summary record say, and is passed over rather than refused. One
/// naming some of them is a cue that does not hold together, and names what it
/// is missing.
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
    if end < start {
        bail!("{output}: a cue row ends at {end} and starts at {start}, so it ends before it starts: {row}");
    }

    Ok(Some(Cue {
        start,
        end,
        text: text.to_string(),
    }))
}

/// The cues gathered so far, in the order their rows arrived.
pub struct Document {
    format: Format,
    cues: Vec<Cue>,
}

impl Document {
    pub fn new(format: Format) -> Document {
        Document {
            format,
            cues: Vec::new(),
        }
    }

    /// Reads one row, keeping the cue it carries and passing over one that is
    /// not a cue.
    pub fn push_row(&mut self, row: &str, output: &str) -> Result<()> {
        if let Some(cue) = read_row(row, output)? {
            self.cues.push(cue);
        }
        Ok(())
    }

    /// The finished document, numbered from one in the order the cues arrived.
    pub fn render(&self) -> String {
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
            out.push_str(&cue.text);
            out.push_str("\n\n");
        }
        out
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

    #[test]
    fn a_row_with_all_three_names_is_a_cue() {
        assert_eq!(
            cue(r#"{"start_t":1.5,"end_t":3.25,"text":"hola"}"#),
            Cue {
                start: 1.5,
                end: 3.25,
                text: "hola".to_string(),
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
        let mut doc = Document::new(Format::Srt);
        doc.push_row(r#"{"start_t":0.0,"end_t":2.0,"text":"uno"}"#, OUT)
            .expect("a cue");
        doc.push_row(r#"{"transcript":"uno dos"}"#, OUT)
            .expect("passed over");
        doc.push_row(r#"{"start_t":2.5,"end_t":4.0,"text":"dos"}"#, OUT)
            .expect("a cue");
        // Two cues, and the record that is not a cue is not one of them.
        assert_eq!(
            doc.render(),
            "1\n00:00:00,000 --> 00:00:02,000\nuno\n\n\
             2\n00:00:02,500 --> 00:00:04,000\ndos\n\n"
        );
    }

    #[test]
    fn webvtt_opens_with_its_header_and_a_full_stop_in_its_timestamps() {
        let mut doc = Document::new(Format::WebVtt);
        doc.push_row(r#"{"start_t":0.0,"end_t":2.0,"text":"uno"}"#, OUT)
            .expect("a cue");
        assert_eq!(
            doc.render(),
            "WEBVTT\n\n1\n00:00:00.000 --> 00:00:02.000\nuno\n\n"
        );
    }
}
