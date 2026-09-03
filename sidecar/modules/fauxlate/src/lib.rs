//! A fake translator, so a pipeline can be tested without a real one.
//!
//! One rule, reached two ways. `translate` is a value function: one string
//! in, the same string out with every word carrying `-a` and `-o` in turn -
//! `"Cue one."` becomes `"Cue-a one-o."`. `translate-rows` runs the identical
//! rule over a cue row's `text` field, leaving `start_t` and `end_t` alone;
//! each row's text is its own call, so the alternation restarts at `-a` on
//! every row the way it restarts on every `translate` call.
//!
//! Punctuation trailing a word stays trailing it: `"word!"` becomes
//! `"word-a!"`, not `"word!-a"`. Whitespace between words, however much of
//! it and whatever kind, is copied through untouched.

// generate_all: the world's interfaces come from another package - ffrwd:av
// - and without it bindgen expects them to have been generated elsewhere.
wit_bindgen::generate!({
    path: ["../../wit", "wit"],
    world: "ffrwd:fauxlate/fauxlate",
    generate_all,
});

use exports::ffrwd::av::rows_module::{Guest as RowsGuest, RowsModuleMeta};
use exports::ffrwd::av::values::{FunctionMeta, Guest as ValuesGuest};
use ffrwd::av::types::Meta;
use serde::{Deserialize, Serialize};

const TRANSLATE: &str = "translate";
const TRANSLATE_PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"text":{"type":"string"}},"required":["text"],"additionalProperties":false}"#;
const TRANSLATE_RESULT_SCHEMA: &str = r#"{"type":"string"}"#;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;
const CUE_SCHEMA: &str = r#"{"type":"object","properties":{"start_t":{"type":"number"},"end_t":{"type":"number"},"text":{"type":"string"}},"required":["start_t","end_t","text"],"additionalProperties":false}"#;

/// One cue: a timed span of text, matching `CUE_SCHEMA`.
#[derive(Deserialize, Serialize)]
struct Cue {
    start_t: f64,
    end_t: f64,
    text: String,
}

/// Whether `c` belongs to a word's core rather than to punctuation trailing
/// it.
fn is_core(c: char) -> bool {
    c.is_alphanumeric()
}

/// `word` split at the start of its trailing run of non-alphanumeric
/// characters - `"one."` as `("one", ".")`, `"Cue"` as `("Cue", "")`.
/// Punctuation inside a word, as in `"example.com"`, is not trailing and
/// stays in the core.
fn split_trailing_punctuation(word: &str) -> (&str, &str) {
    let mut boundary = word.len();
    for (i, c) in word.char_indices().rev() {
        if is_core(c) {
            break;
        }
        boundary = i;
    }
    word.split_at(boundary)
}

/// One word with `suffix` grafted on before whatever punctuation trails it.
fn translate_word(word: &str, suffix: &str) -> String {
    let (core, punctuation) = split_trailing_punctuation(word);
    format!("{core}-{suffix}{punctuation}")
}

/// The word rule: every word in `text` gains `-a` and `-o` in turn, in the
/// order it appears; whitespace is copied through exactly as it arrived.
fn translate(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    let mut suffixes = ["a", "o"].into_iter().cycle();
    let mut chars = text.char_indices().peekable();

    while let Some(&(start, c)) = chars.peek() {
        if c.is_whitespace() {
            let mut end = start;
            while let Some(&(i, c)) = chars.peek() {
                if !c.is_whitespace() {
                    break;
                }
                end = i + c.len_utf8();
                chars.next();
            }
            out.push_str(&text[start..end]);
        } else {
            let mut end = start;
            while let Some(&(i, c)) = chars.peek() {
                if c.is_whitespace() {
                    break;
                }
                end = i + c.len_utf8();
                chars.next();
            }
            out.push_str(&translate_word(&text[start..end], suffixes.next().unwrap()));
        }
    }
    out
}

/// `fauxlate` takes no params - the word rule is fixed - so `init` accepts
/// only the empty object, the same convention `double` and its neighbours
/// use.
fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("fauxlate takes no params, got: {other}")),
    }
}

struct Fauxlate;

impl ValuesGuest for Fauxlate {
    fn list_functions() -> Vec<FunctionMeta> {
        vec![FunctionMeta {
            name: TRANSLATE.to_string(),
            params_schema: TRANSLATE_PARAMS_SCHEMA.to_string(),
            result_schema: TRANSLATE_RESULT_SCHEMA.to_string(),
        }]
    }

    fn invoke(name: String, args: String) -> Result<String, String> {
        if name != TRANSLATE {
            return Err(format!(
                "fauxlate does not export {name}; it exports {TRANSLATE}"
            ));
        }
        let parsed: serde_json::Value = serde_json::from_str(&args)
            .map_err(|e| format!("{TRANSLATE}: args is not valid JSON: {e}"))?;
        let text = parsed
            .as_object()
            .and_then(|o| o.get("text"))
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("{TRANSLATE}: args must be an object with a string \"text\""))?;
        serde_json::to_string(&translate(text))
            .map_err(|e| format!("{TRANSLATE}: serializing result: {e}"))
    }
}

impl RowsGuest for Fauxlate {
    fn describe() -> RowsModuleMeta {
        RowsModuleMeta {
            meta: Meta {
                name: "fauxlate".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: CUE_SCHEMA.to_string(),
                pixel_formats: Vec::new(),
                sample_formats: Vec::new(),
                sample_rates: Vec::new(),
                channel_counts: Vec::new(),
                rows_language: Vec::new(),
            },
            input_rows_schema: CUE_SCHEMA.to_string(),
        }
    }

    fn init(params: String) -> Result<(), String> {
        validate_params(&params)
    }

    fn process(rows: Vec<String>) -> Result<Vec<String>, String> {
        rows.iter()
            .map(|row| {
                let mut cue: Cue = serde_json::from_str(row)
                    .map_err(|e| format!("translate-rows: {row}: not a cue: {e}"))?;
                cue.text = translate(&cue.text);
                serde_json::to_string(&cue)
                    .map_err(|e| format!("translate-rows: serializing a cue: {e}"))
            })
            .collect()
    }

    fn finish() -> Result<Vec<String>, String> {
        // Every row is translated and emitted the moment `process` sees it;
        // nothing is held back for a final call to release.
        Ok(Vec::new())
    }
}

export!(Fauxlate);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_word_gains_a_and_o_in_turn() {
        assert_eq!(translate("Cue one."), "Cue-a one-o.");
    }

    #[test]
    fn a_third_word_returns_to_a() {
        assert_eq!(translate("one two three"), "one-a two-o three-a");
    }

    #[test]
    fn empty_text_stays_empty() {
        assert_eq!(translate(""), "");
    }

    #[test]
    fn a_word_already_ending_in_punctuation_keeps_it_trailing() {
        assert_eq!(translate("Wait!"), "Wait-a!");
        assert_eq!(translate("Wait!!"), "Wait-a!!");
    }

    #[test]
    fn a_number_is_a_word_like_any_other() {
        assert_eq!(translate("3"), "3-a");
        assert_eq!(translate("run 3."), "run-a 3-o.");
    }

    #[test]
    fn whitespace_between_words_is_copied_through_exactly() {
        assert_eq!(translate("one  two\tthree"), "one-a  two-o\tthree-a");
    }

    #[test]
    fn punctuation_inside_a_word_is_not_trailing() {
        assert_eq!(translate("example.com"), "example.com-a");
    }

    #[test]
    fn no_params_is_accepted_and_anything_else_is_refused() {
        assert!(validate_params("").is_ok());
        assert!(validate_params("{}").is_ok());
        let Err(err) = validate_params(r#"{"lang":"es"}"#) else {
            panic!("fauxlate takes no params and should refuse one");
        };
        assert!(err.contains("no params"), "got: {err}");
    }
}
