//! A rows module that puts a vector beside every note it reads.
//!
//! The fixture for a rows module standing between a producer and a packet
//! filter: `note_rows` writes `{"pts": <tick>, "note": "<text>"}`, this reads
//! each one and hands back the same row with an embedding added and the note
//! marked, and a filter downstream weaves what comes out. Rows in, rows out,
//! no stream anywhere, which is the shape a real embedder over a transcript
//! takes.
//!
//! The note gains an `e-` in front of it so the bytes a filter writes say
//! which rows came through here. The vector is eight buckets of the note's
//! own letters, L2-normalized, so two notes sharing letters land closer by
//! cosine than two sharing none, and the same note always counts the same
//! way.

wit_bindgen::generate!({
    path: "../../wit",
    world: "rows-module-host",
});

use exports::ffrwd::av::rows_module::{Guest, Meta, RowsModuleMeta};
use serde::{Deserialize, Serialize};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;
const INPUT_SCHEMA: &str = r#"{"type":"object","properties":{"pts":{"type":"integer"},"note":{"type":"string"}},"required":["pts","note"],"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"pts":{"type":"integer"},"note":{"type":"string"},"vector":{"type":"array","items":{"type":"number"}}},"required":["pts","note","vector"],"additionalProperties":false}"#;

/// What goes in front of a note that has been through here.
const MARK: &str = "e-";

/// How many components one vector carries.
const DIMS: usize = 8;

/// The eight buckets a note's letters are counted into: vowels, then seven
/// consonant groups. Together they partition the alphabet, each letter in
/// exactly one.
const LETTER_GROUPS: [&[char]; DIMS] = [
    &['a', 'e', 'i', 'o', 'u'],
    &['b', 'p', 'm'],
    &['f', 'v', 'w'],
    &['t', 'd', 'n'],
    &['s', 'z', 'c'],
    &['k', 'g', 'q'],
    &['l', 'r', 'y'],
    &['h', 'j', 'x'],
];

/// One row this module reads: a note and the tick it belongs at.
#[derive(Deserialize)]
struct NoteRow {
    pts: i64,
    note: String,
}

/// One row it writes: the same note, marked, with its embedding beside it.
#[derive(Serialize)]
struct EmbeddedRow {
    pts: i64,
    note: String,
    vector: Vec<f64>,
}

/// `text`, lowercased, counted letter by letter into the eight buckets and
/// L2-normalized, so only the letter mix survives and not the length. Text
/// with no letters at all counts to zero in every bucket; normalizing would
/// divide by zero, so that vector is returned as it is.
fn embed(text: &str) -> Vec<f64> {
    let mut counts = vec![0.0_f64; DIMS];
    for c in text.to_lowercase().chars() {
        if let Some(bucket) = LETTER_GROUPS.iter().position(|group| group.contains(&c)) {
            counts[bucket] += 1.0;
        }
    }
    let norm = counts.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm > 0.0 {
        for v in &mut counts {
            *v /= norm;
        }
    }
    counts
}

/// The rule, off the wire and back: one input row becomes one output row.
fn embed_row(row: &str) -> Result<String, String> {
    let read: NoteRow =
        serde_json::from_str(row).map_err(|e| format!("embed_notes: {row}: not a note: {e}"))?;
    let vector = embed(&read.note);
    let written = EmbeddedRow {
        pts: read.pts,
        note: format!("{MARK}{}", read.note),
        vector,
    };
    serde_json::to_string(&written).map_err(|e| format!("embed_notes: serializing a row: {e}"))
}

/// This module has no params - the rule is fixed - so only the empty object
/// is accepted, the convention its neighbours use.
fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("embed_notes takes no params, got: {other}")),
    }
}

struct EmbedNotes;

impl Guest for EmbedNotes {
    fn describe() -> RowsModuleMeta {
        RowsModuleMeta {
            meta: Meta {
                name: "embed_notes".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
                pixel_formats: Vec::new(),
                sample_formats: Vec::new(),
                sample_rates: Vec::new(),
                channel_counts: Vec::new(),
                rows_language: Vec::new(),
            },
            input_rows_schema: INPUT_SCHEMA.to_string(),
        }
    }

    fn init(params: String) -> Result<(), String> {
        validate_params(&params)
    }

    fn process(rows: Vec<String>) -> Result<Vec<String>, String> {
        rows.iter().map(|row| embed_row(row)).collect()
    }

    fn finish() -> Result<Vec<String>, String> {
        // Every row leaves on the call that brought it in; nothing is held.
        Ok(Vec::new())
    }
}

export!(EmbedNotes);

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na = a.iter().map(|v| v * v).sum::<f64>().sqrt();
        let nb = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        dot / (na * nb)
    }

    #[test]
    fn a_row_keeps_its_tick_and_gains_a_marked_note_and_a_vector() {
        let written = embed_row(r#"{"pts":120,"note":"note-0"}"#).expect("a note embeds");
        let read: serde_json::Value = serde_json::from_str(&written).expect("valid JSON");
        assert_eq!(read["pts"], 120);
        assert_eq!(read["note"], "e-note-0");
        assert_eq!(read["vector"].as_array().expect("an array").len(), DIMS);
    }

    #[test]
    fn a_row_that_is_not_a_note_is_refused_by_name() {
        let Err(err) = embed_row(r#"{"pts":1}"#) else {
            panic!("a row with no note should be refused");
        };
        assert!(err.contains("not a note"), "got: {err}");
    }

    #[test]
    fn the_same_note_embeds_to_the_same_vector() {
        assert_eq!(embed("a cat sat on the mat"), embed("a cat sat on the mat"));
    }

    #[test]
    fn notes_sharing_letters_are_closer_than_notes_sharing_none() {
        let cat = embed("cat");
        let bat = embed("bat");
        let xyz = embed("xyz");
        assert!(
            cosine(&cat, &bat) > cosine(&cat, &xyz),
            "cat~bat = {}, cat~xyz = {}",
            cosine(&cat, &bat),
            cosine(&cat, &xyz)
        );
    }

    #[test]
    fn a_note_with_no_letters_embeds_to_the_zero_vector() {
        assert_eq!(embed("123 !?"), vec![0.0; DIMS]);
    }

    #[test]
    fn a_vector_is_l2_normalized() {
        let v = embed("mississippi");
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9, "norm = {norm}");
    }

    #[test]
    fn every_letter_lands_in_exactly_one_bucket() {
        for c in 'a'..='z' {
            let hits = LETTER_GROUPS.iter().filter(|g| g.contains(&c)).count();
            assert_eq!(hits, 1, "letter '{c}' landed in {hits} buckets");
        }
    }

    #[test]
    fn no_params_is_accepted_and_anything_else_is_refused() {
        assert!(validate_params("").is_ok());
        assert!(validate_params("{}").is_ok());
        assert!(validate_params(r#"{"dims":4}"#).is_err());
    }
}
