//! `rowfilter`: the one node of a network the host provides itself.
//!
//! It is spelled like a module - `[a]rowfilter=pred=<json>[b]` - and wired
//! like one, but nothing is compiled and nothing is instantiated. Frames pass
//! through untouched, pixels and all; only the rows travelling with them are
//! kept or dropped, by a predicate written as a JSON tree.
//!
//! The name is reserved: no `-m` binds it, because the host answers for it.

use std::collections::HashSet;

use anyhow::{anyhow, bail, Result};
use ffrwd_wasm_runtime::runtime::{Frame, Shape};
use serde_json::{Map, Value};

/// The node name the grammar reserves.
pub const NODE: &str = "rowfilter";

/// Its only option.
const PRED: &str = "pred";

/// How the host drives it: one frame in, that same frame out.
pub const SHAPE: Shape = Shape {
    window: 1,
    stride: 1,
    pure: true,
    one_to_one: true,
};

/// The operators a predicate is built from, for a refusal listing them.
const OPERATORS: &str = "eq, ne, lt, le, gt, ge, and, or, not";

/// One side of a comparison: a field of the row, or a value written into the
/// predicate.
enum Operand {
    Field(String),
    Lit(Value),
}

/// The six comparisons.
#[derive(Clone, Copy)]
enum Compare {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Compare {
    fn parse(key: &str) -> Option<Compare> {
        Some(match key {
            "eq" => Compare::Eq,
            "ne" => Compare::Ne,
            "lt" => Compare::Lt,
            "le" => Compare::Le,
            "gt" => Compare::Gt,
            "ge" => Compare::Ge,
            _ => return None,
        })
    }

    fn holds(self, ordering: std::cmp::Ordering) -> bool {
        match self {
            Compare::Eq => ordering.is_eq(),
            Compare::Ne => ordering.is_ne(),
            Compare::Lt => ordering.is_lt(),
            Compare::Le => ordering.is_le(),
            Compare::Gt => ordering.is_gt(),
            Compare::Ge => ordering.is_ge(),
        }
    }
}

/// A predicate over one row.
enum Pred {
    Compare(Compare, Operand, Operand),
    And(Vec<Pred>),
    Or(Vec<Pred>),
    Not(Box<Pred>),
}

/// Why a row could not be judged. Either way the row is dropped; a mismatch
/// is worth saying out loud once, an absent field is not.
enum Undecided {
    /// A field the predicate names is not on this row.
    Absent,
    /// The two sides are of different types.
    Mismatch {
        field: Option<String>,
        left: &'static str,
        right: &'static str,
    },
}

/// A JSON value's type, as a refusal or a note spells it.
fn spell_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// One operand: `{"field": "<name>"}` or `{"lit": <value>}`.
fn parse_operand(value: &Value) -> Result<Operand> {
    let bad =
        || anyhow!(r#"{NODE}: {PRED}: an operand is {{"field": "<name>"}} or {{"lit": <value>}}"#);
    let object = value.as_object().ok_or_else(bad)?;
    if object.len() != 1 {
        return Err(bad());
    }
    let (key, argument) = object.iter().next().expect("one entry");
    match key.as_str() {
        "field" => {
            let name = argument.as_str().ok_or_else(bad)?;
            if name.is_empty() {
                bail!("{NODE}: {PRED}: a field operand names an empty field");
            }
            Ok(Operand::Field(name.to_string()))
        }
        "lit" => match argument {
            Value::Bool(_) | Value::Number(_) | Value::String(_) => {
                Ok(Operand::Lit(argument.clone()))
            }
            other => bail!(
                "{NODE}: {PRED}: a literal is a string, a number or a boolean, and this is {}",
                spell_type(other)
            ),
        },
        _ => Err(bad()),
    }
}

/// The two operands a comparison takes.
fn parse_operands(key: &str, argument: &Value) -> Result<(Operand, Operand)> {
    let list = argument.as_array().ok_or_else(|| {
        anyhow!(
            "{NODE}: {PRED}: '{key}' takes two operands as a list, and was given {} instead",
            spell_type(argument)
        )
    })?;
    if list.len() != 2 {
        bail!(
            "{NODE}: {PRED}: '{key}' takes two operands, got {}",
            list.len()
        );
    }
    Ok((parse_operand(&list[0])?, parse_operand(&list[1])?))
}

/// One predicate node, and everything below it.
fn parse_pred(value: &Value) -> Result<Pred> {
    let object = value.as_object().ok_or_else(|| {
        anyhow!(
            "{NODE}: {PRED}: a predicate is a JSON object, and this is {}",
            spell_type(value)
        )
    })?;
    if object.len() != 1 {
        bail!("{NODE}: {PRED} has no operator; a predicate is one of {OPERATORS}");
    }
    let (key, argument) = object.iter().next().expect("one entry");

    if let Some(op) = Compare::parse(key) {
        let (left, right) = parse_operands(key, argument)?;
        return Ok(Pred::Compare(op, left, right));
    }
    match key.as_str() {
        "and" | "or" => {
            let list = argument.as_array().ok_or_else(|| {
                anyhow!(
                    "{NODE}: {PRED}: '{key}' takes a list of predicates, and was given {}",
                    spell_type(argument)
                )
            })?;
            let parsed = list.iter().map(parse_pred).collect::<Result<Vec<_>>>()?;
            Ok(if key == "and" {
                Pred::And(parsed)
            } else {
                Pred::Or(parsed)
            })
        }
        "not" => Ok(Pred::Not(Box::new(parse_pred(argument)?))),
        _ => bail!("{NODE}: {PRED} has no operator; a predicate is one of {OPERATORS}"),
    }
}

/// The value an operand stands for on one row.
fn resolve<'a>(operand: &'a Operand, row: &'a Map<String, Value>) -> Result<&'a Value, Undecided> {
    match operand {
        Operand::Lit(value) => Ok(value),
        Operand::Field(name) => row.get(name).ok_or(Undecided::Absent),
    }
}

/// The field a comparison is about, for the note a mismatch prints. A
/// comparison of two literals is about no field at all.
fn about(left: &Operand, right: &Operand) -> Option<String> {
    match (left, right) {
        (Operand::Field(name), _) | (_, Operand::Field(name)) => Some(name.clone()),
        _ => None,
    }
}

/// Two values compared: numbers numerically, strings lexically, booleans with
/// false below true. Anything else is a mismatch, and the row is dropped.
fn compare(
    op: Compare,
    left: &Value,
    right: &Value,
    field: Option<String>,
) -> Result<bool, Undecided> {
    let ordering = match (left, right) {
        (Value::Number(a), Value::Number(b)) => match (a.as_f64(), b.as_f64()) {
            (Some(a), Some(b)) => a.partial_cmp(&b),
            _ => None,
        },
        (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
        (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
        _ => None,
    };
    match ordering {
        Some(ordering) => Ok(op.holds(ordering)),
        None => Err(Undecided::Mismatch {
            field,
            left: spell_type(left),
            right: spell_type(right),
        }),
    }
}

/// One row judged. An undecidable comparison anywhere drops the row.
fn eval(pred: &Pred, row: &Map<String, Value>) -> Result<bool, Undecided> {
    match pred {
        Pred::Compare(op, left, right) => {
            let a = resolve(left, row)?;
            let b = resolve(right, row)?;
            compare(*op, a, b, about(left, right))
        }
        Pred::And(list) => {
            for pred in list {
                if !eval(pred, row)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        Pred::Or(list) => {
            for pred in list {
                if eval(pred, row)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        Pred::Not(pred) => Ok(!eval(pred, row)?),
    }
}

/// The opened node: its predicate, and which fields it has already complained
/// about.
pub struct RowFilter {
    pred: Pred,
    /// Fields already named on stderr. The note is once per field for the
    /// whole run, because rows are runtime data and a stream that disagrees
    /// about a type is worth noticing once, not once a frame.
    noted: HashSet<String>,
}

impl RowFilter {
    /// Reads a node's options: `pred` is the only one, and it is required.
    pub fn open(options: &[(String, String)]) -> Result<RowFilter> {
        let mut text: Option<&str> = None;
        for (key, value) in options {
            if key != PRED {
                bail!("{NODE} has no option '{key}'; it takes {PRED}=<json>");
            }
            if text.is_some() {
                bail!("{NODE} is given the option '{PRED}' twice");
            }
            text = Some(value);
        }
        let Some(text) = text else {
            bail!("{NODE} takes one option, {PRED}=<json>, and was given none");
        };
        let value: Value = serde_json::from_str(text)
            .map_err(|e| anyhow!("{NODE}: {PRED} is not valid JSON: {e}"))?;
        Ok(RowFilter {
            pred: parse_pred(&value)?,
            noted: HashSet::new(),
        })
    }

    /// One frame through. Its pixels are never read, so the frame moves rather
    /// than being copied; only its rows are judged.
    pub fn pass(&mut self, mut frame: Frame) -> Frame {
        let rows = std::mem::take(&mut frame.rows);
        frame.rows = self.keep(rows);
        frame
    }

    /// The rows the predicate keeps. A row that is not a JSON object, or that
    /// the predicate cannot judge, is dropped.
    pub fn keep(&mut self, rows: Vec<String>) -> Vec<String> {
        rows.into_iter()
            .filter(|row| {
                let Ok(Value::Object(parsed)) = serde_json::from_str::<Value>(row) else {
                    return false;
                };
                match eval(&self.pred, &parsed) {
                    Ok(keep) => keep,
                    Err(Undecided::Absent) => false,
                    Err(mismatch) => {
                        self.note(&mismatch);
                        false
                    }
                }
            })
            .collect()
    }

    /// Names a type mismatch on stderr, once per field.
    fn note(&mut self, mismatch: &Undecided) {
        let Undecided::Mismatch { field, left, right } = mismatch else {
            return;
        };
        let key = field.clone().unwrap_or_default();
        if !self.noted.insert(key) {
            return;
        }
        match field {
            Some(name) => eprintln!(
                "[{NODE}] field '{name}' compares {left} with {right}; those rows are dropped"
            ),
            None => eprintln!(
                "[{NODE}] a comparison of {left} with {right} is never true; those rows are dropped"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open(pred: &str) -> RowFilter {
        RowFilter::open(&[(PRED.to_string(), pred.to_string())]).expect("parses")
    }

    fn rows(items: &[&str]) -> Vec<String> {
        items.iter().map(|r| r.to_string()).collect()
    }

    /// The message a refused node comes back with. An opened one carries a
    /// predicate rather than anything printable, so the refusal is unwrapped
    /// here instead of by `expect_err`.
    fn refuse(options: &[(&str, &str)]) -> String {
        let options: Vec<(String, String)> = options
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        match RowFilter::open(&options) {
            Ok(_) => panic!("opened, and this predicate must be refused"),
            Err(error) => error.to_string(),
        }
    }

    #[test]
    fn a_numeric_comparison_keeps_the_rows_above_the_bound() {
        let mut filter = open(r#"{"ge":[{"field":"shot"},{"lit":1}]}"#);
        let kept = filter.keep(rows(&[r#"{"shot":0}"#, r#"{"shot":1}"#, r#"{"shot":2}"#]));
        assert_eq!(kept, rows(&[r#"{"shot":1}"#, r#"{"shot":2}"#]));
    }

    #[test]
    fn numbers_compare_numerically_rather_than_as_text() {
        let mut filter = open(r#"{"gt":[{"field":"n"},{"lit":9}]}"#);
        // As text "10" sorts below "9"; as numbers it does not.
        assert_eq!(filter.keep(rows(&[r#"{"n":10}"#])), rows(&[r#"{"n":10}"#]));
    }

    #[test]
    fn strings_compare_lexically() {
        let mut filter = open(r#"{"eq":[{"field":"class"},{"lit":"person"}]}"#);
        let kept = filter.keep(rows(&[r#"{"class":"person"}"#, r#"{"class":"bus"}"#]));
        assert_eq!(kept, rows(&[r#"{"class":"person"}"#]));
    }

    #[test]
    fn and_or_and_not_compose() {
        let mut filter = open(
            r#"{"and":[{"or":[{"eq":[{"field":"c"},{"lit":"a"}]},{"eq":[{"field":"c"},{"lit":"b"}]}]},{"not":{"lt":[{"field":"s"},{"lit":0.5}]}}]}"#,
        );
        let kept = filter.keep(rows(&[
            r#"{"c":"a","s":0.9}"#,
            r#"{"c":"a","s":0.1}"#,
            r#"{"c":"b","s":0.6}"#,
            r#"{"c":"z","s":0.9}"#,
        ]));
        assert_eq!(
            kept,
            rows(&[r#"{"c":"a","s":0.9}"#, r#"{"c":"b","s":0.6}"#])
        );
    }

    #[test]
    fn an_empty_and_keeps_everything_and_an_empty_or_keeps_nothing() {
        assert_eq!(
            open(r#"{"and":[]}"#).keep(rows(&[r#"{"a":1}"#])),
            rows(&[r#"{"a":1}"#])
        );
        assert!(open(r#"{"or":[]}"#).keep(rows(&[r#"{"a":1}"#])).is_empty());
    }

    #[test]
    fn a_row_lacking_the_field_is_dropped_silently() {
        let mut filter = open(r#"{"eq":[{"field":"class"},{"lit":"person"}]}"#);
        // The alien row is skipped the way every consumer skips one.
        let kept = filter.keep(rows(&[r#"{"class":"person"}"#, r#"{"shot":3}"#]));
        assert_eq!(kept, rows(&[r#"{"class":"person"}"#]));
        assert!(filter.noted.is_empty(), "an absent field is not a mismatch");
    }

    #[test]
    fn a_cross_type_comparison_drops_the_row_and_names_the_field_once() {
        let mut filter = open(r#"{"gt":[{"field":"score"},{"lit":0.5}]}"#);
        let kept = filter.keep(rows(&[
            r#"{"score":"high"}"#,
            r#"{"score":"low"}"#,
            r#"{"score":0.9}"#,
        ]));
        assert_eq!(kept, rows(&[r#"{"score":0.9}"#]));
        assert_eq!(
            filter.noted.len(),
            1,
            "two bad rows on one field is one note, not two"
        );
    }

    #[test]
    fn a_row_that_is_not_an_object_is_dropped() {
        let mut filter = open(r#"{"eq":[{"field":"a"},{"lit":1}]}"#);
        assert!(filter.keep(rows(&["[1,2]", "not json", "7"])).is_empty());
    }

    #[test]
    fn a_frame_keeps_its_pixels_and_loses_only_its_rows() {
        let mut filter = open(r#"{"eq":[{"field":"a"},{"lit":1}]}"#);
        let frame = Frame {
            pts: 7,
            data: vec![1, 2, 3, 4].into(),
            rows: rows(&[r#"{"a":1}"#, r#"{"a":2}"#]),
        };
        let out = filter.pass(frame);
        assert_eq!(out.pts, 7);
        assert_eq!(*out.data, vec![1, 2, 3, 4], "the pixels are untouched");
        assert_eq!(out.rows, rows(&[r#"{"a":1}"#]));
    }

    #[test]
    fn a_node_with_no_pred_is_refused_naming_the_option() {
        let message = refuse(&[]);
        assert!(message.contains(NODE), "got: {message}");
        assert!(message.contains("pred=<json>"), "got: {message}");
    }

    #[test]
    fn an_option_that_is_not_pred_is_refused_by_name() {
        let message = refuse(&[("threshold", "1")]);
        assert!(message.contains("'threshold'"), "got: {message}");
    }

    #[test]
    fn a_pred_that_is_not_json_is_refused_naming_the_node() {
        let message = refuse(&[(PRED, "{\"eq\":")]);
        assert!(
            message.starts_with("rowfilter: pred is not valid JSON"),
            "got: {message}"
        );
    }

    #[test]
    fn an_unknown_operator_is_refused_listing_the_operators() {
        let message = refuse(&[(PRED, r#"{"between":[1,2]}"#)]);
        assert!(message.contains("has no operator"), "got: {message}");
        assert!(message.contains("and, or, not"), "got: {message}");
    }

    #[test]
    fn a_comparison_of_three_operands_is_refused_with_the_count() {
        let message = refuse(&[(PRED, r#"{"eq":[{"lit":1},{"lit":2},{"lit":3}]}"#)]);
        assert!(
            message.contains("takes two operands, got 3"),
            "got: {message}"
        );
    }

    #[test]
    fn an_operand_that_is_neither_field_nor_lit_is_refused_showing_both() {
        let message = refuse(&[(PRED, r#"{"eq":[{"column":"a"},{"lit":1}]}"#)]);
        assert!(message.contains(r#""field""#), "got: {message}");
        assert!(message.contains(r#""lit""#), "got: {message}");
    }
}
