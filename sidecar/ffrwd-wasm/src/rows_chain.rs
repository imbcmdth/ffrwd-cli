//! A chain of rows modules riding alongside a producer, each one `-m <path>
//! -rows-from <index>` on the argv named. The rows a rows-bearing output
//! would have written - `-f ndjson`, `-f srt` or `-f webvtt` - flow through
//! every hop first, in argv order, and what the last hop returns is what the
//! output actually gets.
//!
//! Each hop only reads the rows shaped like the ones its own module declared
//! (`input_rows_schema`); a row of another shape - a trailing summary riding
//! beside the ones a module actually reads, say - passes the hop untouched
//! rather than being handed to it.

use anyhow::Result;
use ffrwd_wasm_runtime::runtime::{self, RowsModule};

/// Whether `row` is shaped the way `schema` declares, well enough to enter a
/// rows module's `process`: every field `schema`'s `"required"` names is on
/// the row. This is not a JSON Schema checker - a rows module's own schema
/// only ever uses `"required"` to tell its rows from a shape riding the same
/// output beside them (a trailing count record, say), so that is the one
/// rule read here. A schema naming none is read as accepting every row.
pub fn row_matches_schema(row: &str, schema: &serde_json::Value) -> bool {
    let Ok(serde_json::Value::Object(object)) = serde_json::from_str::<serde_json::Value>(row)
    else {
        return false;
    };
    match schema.get("required").and_then(|r| r.as_array()) {
        Some(required) => required
            .iter()
            .all(|name| name.as_str().is_some_and(|name| object.contains_key(name))),
        None => true,
    }
}

/// One hop of the chain: the module, and the schema of the row it reads.
struct Hop {
    module: RowsModule,
    input_schema: serde_json::Value,
}

impl Hop {
    fn open(path: &str) -> Result<Hop> {
        let described = runtime::describe_rows_module(path)?;
        let input_schema: serde_json::Value = if described.input_rows_schema.trim().is_empty() {
            serde_json::Value::Object(serde_json::Map::new())
        } else {
            serde_json::from_str(&described.input_rows_schema)?
        };
        let module = RowsModule::open(path, "{}")?;
        Ok(Hop {
            module,
            input_schema,
        })
    }

    /// Routes `rows` through this hop: rows shaped like this module's own
    /// are handed to `process`, the rest ride through untouched. When
    /// `process` hands back as many rows as it read, each transformed row
    /// takes the exact place of the one it came from, so the untouched
    /// rows keep their original position exactly; when it does not - the
    /// module is holding some back for a later call - the transformed rows
    /// are appended after the untouched ones for this call instead, which is
    /// the closest order this host can promise across an uneven batch.
    fn step(&mut self, rows: Vec<String>) -> Result<Vec<String>> {
        let mut slots: Vec<Option<String>> = Vec::with_capacity(rows.len());
        let mut positions: Vec<usize> = Vec::new();
        let mut matching: Vec<String> = Vec::new();
        for row in rows {
            if row_matches_schema(&row, &self.input_schema) {
                positions.push(slots.len());
                matching.push(row);
                slots.push(None);
            } else {
                slots.push(Some(row));
            }
        }
        let transformed = self.module.process(&matching)?;
        if transformed.len() == positions.len() {
            for (position, row) in positions.into_iter().zip(transformed) {
                slots[position] = Some(row);
            }
            Ok(slots
                .into_iter()
                .map(|row| row.expect("every slot was filled"))
                .collect())
        } else {
            let mut out: Vec<String> = slots.into_iter().flatten().collect();
            out.extend(transformed);
            Ok(out)
        }
    }
}

/// The chain a rows-bearing output's rows flow through before they are
/// written, one hop per `-m <path> -rows-from <index>` on the line, in the
/// order they were given.
pub struct RowsChain {
    hops: Vec<Hop>,
}

impl RowsChain {
    /// Opens every hop's module, in `paths`' order - the same order their
    /// `-m -rows-from` pairs were given in, each one's `-rows-from` already
    /// checked to name the hop before it (or the producer, for the first).
    pub fn open(paths: &[String]) -> Result<RowsChain> {
        let hops = paths
            .iter()
            .map(|path| Hop::open(path))
            .collect::<Result<Vec<_>>>()?;
        Ok(RowsChain { hops })
    }

    /// One batch of the producer's rows through every hop in turn.
    pub fn process(&mut self, rows: Vec<String>) -> Result<Vec<String>> {
        let mut current = rows;
        for hop in &mut self.hops {
            current = hop.step(current)?;
        }
        Ok(current)
    }

    /// Every hop's `finish`, called once each after the last `process`: hop
    /// 0's trailing rows flow through hops 1.. the same way an ordinary
    /// batch does, hop 1's trailing rows flow through hops 2.. and so on, and
    /// each hop's resulting rows are appended to the output in that order.
    pub fn finish(&mut self) -> Result<Vec<String>> {
        let mut appended = Vec::new();
        for i in 0..self.hops.len() {
            let mut current = self.hops[i].module.finish()?;
            for hop in &mut self.hops[i + 1..] {
                current = hop.step(current)?;
            }
            appended.extend(current);
        }
        Ok(appended)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cue_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "start_t": {"type": "number"},
                "end_t": {"type": "number"},
                "text": {"type": "string"}
            },
            "required": ["start_t", "end_t", "text"],
            "additionalProperties": false
        })
    }

    #[test]
    fn a_row_carrying_every_required_field_matches() {
        let row = r#"{"start_t":0.0,"end_t":1.5,"text":"hola","pts":3,"time":0.5}"#;
        assert!(
            row_matches_schema(row, &cue_schema()),
            "extra host-stamped fields (pts, time) do not stop a match"
        );
    }

    #[test]
    fn a_trailing_summary_row_does_not_match_the_cue_schema() {
        // captions' own trailing record, `{"cues": n}`: the other arm of its
        // rows schema, and not shaped like the cue rows a consumer reads.
        assert!(!row_matches_schema(r#"{"cues":5}"#, &cue_schema()));
    }

    #[test]
    fn a_row_missing_a_required_field_does_not_match() {
        assert!(!row_matches_schema(
            r#"{"start_t":0.0,"text":"hola"}"#,
            &cue_schema()
        ));
    }

    #[test]
    fn a_schema_with_no_required_list_matches_every_object() {
        let schema = serde_json::json!({"type": "object"});
        assert!(row_matches_schema(r#"{"anything":1}"#, &schema));
    }

    #[test]
    fn a_row_that_is_not_json_never_matches() {
        assert!(!row_matches_schema("not json", &cue_schema()));
    }
}
