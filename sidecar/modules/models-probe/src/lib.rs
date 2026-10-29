//! Reads one pinned file through `ffrwd:models/files` and answers what it
//! got, small enough to assert on by hand.
//!
//! `read` takes `{"name": ...}` and answers the byte length and sha256 of
//! what `files.read` returned, beside the names `files.names` lists. A name
//! nothing binds is the host's Err, passed on as this module's own.

wit_bindgen::generate!({
    path: ["../../wit", "../../wit-models", "wit"],
    // Fully qualified: several packages are in scope, and each has worlds.
    world: "ffrwd:models-probe/models-probe",
    generate_all,
});

use exports::ffrwd::av::values::{FunctionMeta, Guest};
use ffrwd::models::files;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};

const READ: &str = "read";
const READ_PARAMS: &str = r#"{"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":false}"#;
const READ_RESULT: &str = r#"{"type":"object","properties":{"len":{"type":"integer"},"sha256":{"type":"string"},"names":{"type":"array","items":{"type":"string"}}},"required":["len","sha256","names"]}"#;

#[derive(Deserialize)]
struct ReadArgs {
    name: String,
}

fn read(args: &str) -> Result<String, String> {
    let args: ReadArgs = serde_json::from_str(args).map_err(|e| format!("invalid args: {e}"))?;
    let bytes = files::read(&args.name)?;
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    Ok(json!({ "len": bytes.len(), "sha256": digest, "names": files::names() }).to_string())
}

struct ModelsProbe;

impl Guest for ModelsProbe {
    fn list_functions() -> Vec<FunctionMeta> {
        vec![FunctionMeta {
            name: READ.to_string(),
            params_schema: READ_PARAMS.to_string(),
            result_schema: READ_RESULT.to_string(),
        }]
    }

    fn invoke(name: String, args: String) -> Result<String, String> {
        match name.as_str() {
            READ => read(&args),
            other => Err(format!(
                "models-probe does not export {other}; it exports {READ}"
            )),
        }
    }
}

export!(ModelsProbe);
