//! A values-world stand-in for `RETURNS source`: `files` turns a
//! comma-separated list of paths into rows the compiler can bind as
//! rendition rows, each one naming a path ffmpeg opens with `-i` itself.
//! Unlike `source-replay`, this module carries no stream of its own - it
//! runs once at compile time and hands back JSON, the way any other
//! `values` function does.

wit_bindgen::generate!({
    path: "../../wit",
    world: "values-module",
});

use exports::ffrwd::av::values::{FunctionMeta, Guest};
use serde_json::json;

struct SourceFiles;

const FILES: &str = "files";

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"paths":{"type":"string"}},"required":["paths"],"additionalProperties":false}"#;
const RESULT_SCHEMA: &str = r#"{"type":"object","properties":{"rows":{"type":"array","items":{"type":"object","properties":{"url":{"type":"string"},"sequence":{"type":"integer"}},"required":["url"]}},"bounded":{"type":"boolean"}},"required":["rows","bounded"]}"#;

/// `paths` split on commas and trimmed. Refuses an empty list (nothing but
/// whitespace) and an empty entry (two commas with nothing, or nothing,
/// between them) alike - both name zero files where one row needs at least
/// one.
fn parse_paths(paths: &str) -> Result<Vec<&str>, String> {
    if paths.trim().is_empty() {
        return Err(format!("{FILES}: \"paths\" names no files"));
    }
    paths
        .split(',')
        .enumerate()
        .map(|(i, raw)| {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                Err(format!(
                    "{FILES}: \"paths\" has an empty entry at position {}",
                    i + 1
                ))
            } else {
                Ok(trimmed)
            }
        })
        .collect()
}

/// `args` must be a JSON object with a string `paths` key; no others.
fn parse_args(args: &str) -> Result<String, String> {
    let parsed: serde_json::Value =
        serde_json::from_str(args).map_err(|e| format!("{FILES}: args is not valid JSON: {e}"))?;
    let object = parsed
        .as_object()
        .ok_or_else(|| format!("{FILES}: args must be a JSON object, got {parsed}"))?;
    object
        .get("paths")
        .ok_or_else(|| format!("{FILES}: missing required key \"paths\""))?
        .as_str()
        .ok_or_else(|| format!("{FILES}: \"paths\" must be a string"))
        .map(str::to_string)
}

fn files(args: &str) -> Result<String, String> {
    let paths = parse_args(args)?;
    let entries = parse_paths(&paths)?;
    let rows: Vec<_> = entries
        .iter()
        .enumerate()
        .map(|(i, url)| json!({ "url": url, "sequence": (i + 1) as u64 }))
        .collect();
    Ok(json!({ "rows": rows, "bounded": true }).to_string())
}

impl Guest for SourceFiles {
    fn list_functions() -> Vec<FunctionMeta> {
        vec![FunctionMeta {
            name: FILES.to_string(),
            params_schema: PARAMS_SCHEMA.to_string(),
            result_schema: RESULT_SCHEMA.to_string(),
        }]
    }

    fn invoke(name: String, args: String) -> Result<String, String> {
        if name != FILES {
            return Err(format!(
                "source-files does not export {name}; it exports {FILES}"
            ));
        }
        files(&args)
    }
}

export!(SourceFiles);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_trims_and_numbers_rows() {
        let out = files(r#"{"paths": "a.mp4, b.mp4"}"#).expect("ok");
        let value: serde_json::Value = serde_json::from_str(&out).expect("valid json");
        assert_eq!(
            value,
            json!({
                "rows": [
                    {"url": "a.mp4", "sequence": 1},
                    {"url": "b.mp4", "sequence": 2},
                ],
                "bounded": true,
            })
        );
    }

    #[test]
    fn refuses_an_empty_list() {
        assert!(files(r#"{"paths": "   "}"#).is_err());
    }

    #[test]
    fn refuses_an_empty_entry() {
        assert!(files(r#"{"paths": "a.mp4,,b.mp4"}"#).is_err());
    }

    #[test]
    fn refuses_an_unknown_function_name() {
        assert!(SourceFiles::invoke("nope".to_string(), "{}".to_string()).is_err());
    }
}
