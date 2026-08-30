wit_bindgen::generate!({
    path: "../../wit",
    world: "values-module",
});

use exports::ffrwd::av::values::{FunctionMeta, Guest};

struct Brand;

const APPEND_BRAND: &str = "append-brand";

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"title":{"type":"string"},"suffix":{"type":"string"}},"required":["title","suffix"],"additionalProperties":false}"#;
const RESULT_SCHEMA: &str = r#"{"type":"string"}"#;

/// `args` must be a JSON object with string `title` and `suffix` keys; no
/// others. Returns the two fields on success.
fn parse_args(args: &str) -> Result<(String, String), String> {
    let parsed: serde_json::Value = serde_json::from_str(args)
        .map_err(|e| format!("{APPEND_BRAND}: args is not valid JSON: {e}"))?;
    let object = parsed
        .as_object()
        .ok_or_else(|| format!("{APPEND_BRAND}: args must be a JSON object, got {parsed}"))?;

    let title = object
        .get("title")
        .ok_or_else(|| format!("{APPEND_BRAND}: missing required key \"title\""))?
        .as_str()
        .ok_or_else(|| format!("{APPEND_BRAND}: \"title\" must be a string"))?
        .to_string();
    let suffix = object
        .get("suffix")
        .ok_or_else(|| format!("{APPEND_BRAND}: missing required key \"suffix\""))?
        .as_str()
        .ok_or_else(|| format!("{APPEND_BRAND}: \"suffix\" must be a string"))?
        .to_string();

    Ok((title, suffix))
}

impl Guest for Brand {
    fn list_functions() -> Vec<FunctionMeta> {
        vec![FunctionMeta {
            name: APPEND_BRAND.to_string(),
            params_schema: PARAMS_SCHEMA.to_string(),
            result_schema: RESULT_SCHEMA.to_string(),
        }]
    }

    fn invoke(name: String, args: String) -> Result<String, String> {
        if name != APPEND_BRAND {
            return Err(format!(
                "brand does not export {name}; it exports {APPEND_BRAND}"
            ));
        }
        let (title, suffix) = parse_args(&args)?;
        serde_json::to_string(&format!("{title}{suffix}"))
            .map_err(|e| format!("{APPEND_BRAND}: serializing result: {e}"))
    }
}

export!(Brand);
