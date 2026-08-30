//! A value-function module of the world that introduced them, so the adapter
//! carrying that world's `values` interface is exercised.

wit_bindgen::generate!({
    path: "../../worlds/0.3.0",
    world: "values-module",
});

use exports::ffrwd::av::values::{FunctionMeta, Guest};

const WORLD_OF: &str = "world-of";

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;
const RESULT_SCHEMA: &str = r#"{"type":"string"}"#;

struct Adapted030;

impl Guest for Adapted030 {
    fn list_functions() -> Vec<FunctionMeta> {
        vec![FunctionMeta {
            name: WORLD_OF.to_string(),
            params_schema: PARAMS_SCHEMA.to_string(),
            result_schema: RESULT_SCHEMA.to_string(),
        }]
    }

    fn invoke(name: String, _args: String) -> Result<String, String> {
        if name != WORLD_OF {
            return Err(format!(
                "adapted-030 does not export {name}; it exports {WORLD_OF}"
            ));
        }
        Ok("\"ffrwd:av@0.3.0\"".to_string())
    }
}

export!(Adapted030);
