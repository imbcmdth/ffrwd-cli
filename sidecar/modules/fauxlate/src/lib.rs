//! A fake translator and a fake embedder, so a pipeline can be tested
//! without a real one of either.
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
//!
//! `embed_text` is a third export, a value function `RETURNS vector`: an
//! 8-dimensional stand-in for a real embedding, built from nothing more
//! than the mix of letters in its argument. See `embed_text` for the rule.

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

const EMBED_TEXT: &str = "embed_text";
const EMBED_TEXT_PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"prompt":{"type":"string"}},"required":["prompt"],"additionalProperties":false}"#;
const EMBED_TEXT_RESULT_SCHEMA: &str = r#"{"type":"array","items":{"type":"number"}}"#;

/// The eight buckets `embed_text` counts into: vowels, then seven
/// consonant groups grouped by rough place/manner of articulation. Together
/// the eight groups partition the whole alphabet, each letter in exactly
/// one.
const LETTER_GROUPS: [&[char]; 8] = [
    &['a', 'e', 'i', 'o', 'u'],
    &['b', 'p', 'm'],
    &['f', 'v', 'w'],
    &['t', 'd', 'n'],
    &['s', 'z', 'c'],
    &['k', 'g', 'q'],
    &['l', 'r', 'y'],
    &['h', 'j', 'x'],
];

/// A fake embedding standing in for a real one: `text`, lowercased, is
/// counted letter by letter into `LETTER_GROUPS`'s eight buckets, then
/// L2-normalized so only the text's letter *mix* survives, not its length.
/// Same text always counts to the same vector; two texts that share letters
/// land closer together by cosine than two that share none, because they
/// fill the same buckets. Text with no letters at all - `""`, or a string
/// of digits and punctuation - has nothing to count, so every bucket stays
/// zero; normalizing would divide by zero, so the zero vector is returned
/// as is rather than manufactured direction from nothing.
fn embed_text(text: &str) -> [f64; 8] {
    let mut counts = [0.0_f64; 8];
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
        vec![
            FunctionMeta {
                name: TRANSLATE.to_string(),
                params_schema: TRANSLATE_PARAMS_SCHEMA.to_string(),
                result_schema: TRANSLATE_RESULT_SCHEMA.to_string(),
            },
            FunctionMeta {
                name: EMBED_TEXT.to_string(),
                params_schema: EMBED_TEXT_PARAMS_SCHEMA.to_string(),
                result_schema: EMBED_TEXT_RESULT_SCHEMA.to_string(),
            },
        ]
    }

    fn invoke(name: String, args: String) -> Result<String, String> {
        match name.as_str() {
            TRANSLATE => {
                let text = extract_string_arg(TRANSLATE, &args, "text")?;
                serde_json::to_string(&translate(&text))
                    .map_err(|e| format!("{TRANSLATE}: serializing result: {e}"))
            }
            EMBED_TEXT => {
                let prompt = extract_string_arg(EMBED_TEXT, &args, "prompt")?;
                serde_json::to_string(&embed_text(&prompt).to_vec())
                    .map_err(|e| format!("{EMBED_TEXT}: serializing result: {e}"))
            }
            other => Err(format!(
                "fauxlate does not export {other}; it exports {TRANSLATE}, {EMBED_TEXT}"
            )),
        }
    }
}

/// `args`' one string field named `field`: `translate`'s `"text"`,
/// `embed_text`'s `"prompt"`.
fn extract_string_arg(name: &str, args: &str, field: &str) -> Result<String, String> {
    let parsed: serde_json::Value =
        serde_json::from_str(args).map_err(|e| format!("{name}: args is not valid JSON: {e}"))?;
    parsed
        .as_object()
        .and_then(|o| o.get(field))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("{name}: args must be an object with a string \"{field}\""))
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

    fn cosine(a: [f64; 8], b: [f64; 8]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na = a.iter().map(|v| v * v).sum::<f64>().sqrt();
        let nb = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        dot / (na * nb)
    }

    #[test]
    fn the_same_text_embeds_to_the_same_vector() {
        assert_eq!(
            embed_text("a cat sat on the mat"),
            embed_text("a cat sat on the mat")
        );
    }

    #[test]
    fn texts_sharing_letters_are_closer_than_texts_sharing_none() {
        let cat = embed_text("cat");
        let bat = embed_text("bat"); // shares "at" with "cat"
        let xyz = embed_text("xyz"); // shares no letter with "cat"
        assert!(
            cosine(cat, bat) > cosine(cat, xyz),
            "cat~bat = {}, cat~xyz = {}",
            cosine(cat, bat),
            cosine(cat, xyz)
        );
    }

    #[test]
    fn text_with_no_letters_embeds_to_the_zero_vector() {
        assert_eq!(embed_text(""), [0.0; 8]);
        assert_eq!(embed_text("123 !?"), [0.0; 8]);
    }

    #[test]
    fn a_vector_is_l2_normalized() {
        let v = embed_text("mississippi");
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
}
