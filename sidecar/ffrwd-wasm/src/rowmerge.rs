//! `rowmerge`: the host's second node, beside `rowfilter`.
//!
//! It is spelled like a module - `[a]rowmerge=max_distance=<number>[b]` - and
//! wired like one, but nothing is compiled and nothing is instantiated.
//! Frames pass through untouched; the rows travelling with them are collapsed
//! into runs, one row per run.
//!
//! The name is reserved: no `-m` binds it, because the host answers for it.

use anyhow::{anyhow, bail, Result};
use ffrwd_wasm_runtime::runtime::{Frame, Shape};
use serde_json::{Map, Value};

/// The node name the grammar reserves.
pub const NODE: &str = "rowmerge";

/// Its only option.
const MAX_DISTANCE: &str = "max_distance";

/// The two fields a row has to carry to take part in a run.
const START: &str = "start_t";
const END: &str = "end_t";

/// The one field a run joins rather than taking the first row's.
const TEXT: &str = "text";

/// How the host drives it: one frame in, that same frame out.
pub const SHAPE: Shape = Shape {
    window: 1,
    stride: 1,
    pure: true,
    one_to_one: true,
};

/// One row's span, when it carries one.
fn span(row: &Map<String, Value>) -> Option<(f64, f64)> {
    let start = row.get(START)?.as_f64()?;
    let end = row.get(END)?.as_f64()?;
    Some((start, end))
}

/// The rows of one run joined into the row that stands for it: the first
/// row's fields, its `text` joined with one space, and the furthest end any
/// of them reached.
fn join(run: &[Map<String, Value>], end: f64) -> Map<String, Value> {
    let mut merged = run[0].clone();
    if merged.contains_key(TEXT) {
        let text = run
            .iter()
            .filter_map(|row| row.get(TEXT).and_then(|value| value.as_str()))
            .collect::<Vec<_>>()
            .join(" ");
        merged.insert(TEXT.to_string(), Value::String(text));
    }
    if let Some(number) = serde_json::Number::from_f64(end) {
        merged.insert(END.to_string(), Value::Number(number));
    }
    merged
}

/// The run being held: its rows, and the furthest end they reached.
struct Run {
    rows: Vec<Map<String, Value>>,
    end: f64,
}

/// The merge itself, one row at a time.
///
/// A row whose `start_t` is no more than `max_distance` past the run's end
/// joins it; anything further away closes the run and starts a new one. Rows
/// are taken in the order they arrive, which is `start_t` order for a stream
/// and is what [`merge`] sorts a whole array into.
pub struct Merger {
    max_distance: f64,
    run: Option<Run>,
}

impl Merger {
    pub fn new(max_distance: f64) -> Merger {
        Merger {
            max_distance,
            run: None,
        }
    }

    /// One row in; the row a closed run left behind, if this one closed it.
    pub fn push(&mut self, row: Map<String, Value>, start: f64, end: f64) -> Option<Value> {
        match &mut self.run {
            Some(run) if start - run.end <= self.max_distance => {
                run.rows.push(row);
                run.end = run.end.max(end);
                None
            }
            _ => {
                let closed = self
                    .run
                    .take()
                    .map(|run| Value::Object(join(&run.rows, run.end)));
                self.run = Some(Run {
                    rows: vec![row],
                    end,
                });
                closed
            }
        }
    }

    /// The run still open, if any. Called once the rows have run out.
    pub fn flush(&mut self) -> Option<Value> {
        self.run
            .take()
            .map(|run| Value::Object(join(&run.rows, run.end)))
    }
}

/// The opened node: the distance rows merge across, and the run it holds.
pub struct RowMerge {
    merger: Merger,
}

impl RowMerge {
    /// Reads a node's options: `max_distance` is the only one, and it is
    /// required.
    pub fn open(options: &[(String, String)]) -> Result<RowMerge> {
        let mut text: Option<&str> = None;
        for (key, value) in options {
            if key != MAX_DISTANCE {
                bail!("{NODE} has no option '{key}'; it takes {MAX_DISTANCE}=<number>");
            }
            if text.is_some() {
                bail!("{NODE} is given the option '{MAX_DISTANCE}' twice");
            }
            text = Some(value);
        }
        let Some(text) = text else {
            bail!("{NODE} takes one option, {MAX_DISTANCE}=<number>, and was given none");
        };
        let max_distance: f64 = text
            .parse()
            .map_err(|_| anyhow!("{NODE}: {MAX_DISTANCE} is not a number: {text}"))?;
        if !max_distance.is_finite() || max_distance < 0.0 {
            bail!("{NODE}: {MAX_DISTANCE} is a distance in seconds and cannot be {text}");
        }
        Ok(RowMerge {
            merger: Merger::new(max_distance),
        })
    }

    /// One frame through. Its pixels are never read, so the frame moves rather
    /// than being copied; it carries whichever runs closed while it went by.
    pub fn pass(&mut self, mut frame: Frame) -> Frame {
        let rows = std::mem::take(&mut frame.rows);
        frame.rows = self.merged(rows);
        frame
    }

    /// The rows these rows collapse into. A row that is not a JSON object, or
    /// that carries no span, rides through untouched: it is not a row this
    /// node was written for, the same posture every other consumer takes.
    pub fn merged(&mut self, rows: Vec<String>) -> Vec<String> {
        let mut out = Vec::new();
        for row in rows {
            let Ok(Value::Object(parsed)) = serde_json::from_str::<Value>(&row) else {
                out.push(row);
                continue;
            };
            match span(&parsed) {
                Some((start, end)) => {
                    if let Some(closed) = self.merger.push(parsed, start, end) {
                        out.push(closed.to_string());
                    }
                }
                None => out.push(row),
            }
        }
        out
    }

    /// The run still open once the stream has ended.
    pub fn finish(&mut self) -> Vec<String> {
        self.merger
            .flush()
            .map(|row| vec![row.to_string()])
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A whole array of rows merged at once: `start_t` order, then the one
    /// merge a stream gets. The rows a stream carries already arrive in that
    /// order, so this is the streaming stage with the sort in front of it.
    fn merge(rows: Vec<Map<String, Value>>, max_distance: f64) -> Vec<Value> {
        let mut spanned: Vec<(f64, f64, Map<String, Value>)> = rows
            .into_iter()
            .filter_map(|row| span(&row).map(|(start, end)| (start, end, row)))
            .collect();
        spanned.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut merger = Merger::new(max_distance);
        let mut out = Vec::new();
        for (start, end, row) in spanned {
            out.extend(merger.push(row, start, end));
        }
        out.extend(merger.flush());
        out
    }

    /// The merge table, shared with the compiler's own: each case is the rows
    /// in, the distance, and the spans out.
    fn merged(rows: &[&str], max_distance: f64) -> Vec<String> {
        let parsed = rows
            .iter()
            .map(|row| match serde_json::from_str::<Value>(row) {
                Ok(Value::Object(object)) => object,
                _ => panic!("a case row is a JSON object"),
            })
            .collect();
        merge(parsed, max_distance)
            .into_iter()
            .map(|row| row.to_string())
            .collect()
    }

    fn spans(rows: &[String]) -> Vec<(f64, f64)> {
        rows.iter()
            .map(|row| {
                let value: Value = serde_json::from_str(row).expect("json");
                (
                    value[START].as_f64().expect("start_t"),
                    value[END].as_f64().expect("end_t"),
                )
            })
            .collect()
    }

    #[test]
    fn an_empty_array_merges_to_an_empty_array() {
        assert!(merged(&[], 0.0).is_empty());
    }

    #[test]
    fn one_row_is_its_own_run() {
        let out = merged(&[r#"{"start_t":1.0,"end_t":2.0}"#], 0.0);
        assert_eq!(spans(&out), vec![(1.0, 2.0)]);
    }

    #[test]
    fn touching_rows_merge_at_distance_zero() {
        let out = merged(
            &[
                r#"{"start_t":0.0,"end_t":1.0}"#,
                r#"{"start_t":1.0,"end_t":2.0}"#,
            ],
            0.0,
        );
        assert_eq!(spans(&out), vec![(0.0, 2.0)]);
    }

    #[test]
    fn a_gap_equal_to_max_distance_merges_and_a_wider_one_does_not() {
        let rows = [
            r#"{"start_t":0.0,"end_t":1.0}"#,
            r#"{"start_t":2.0,"end_t":3.0}"#,
        ];
        assert_eq!(spans(&merged(&rows, 1.0)), vec![(0.0, 3.0)]);
        assert_eq!(
            spans(&merged(&rows, 0.9)),
            vec![(0.0, 1.0), (2.0, 3.0)],
            "a gap past the distance keeps two rows"
        );
    }

    #[test]
    fn overlapping_rows_merge_whatever_the_distance() {
        let out = merged(
            &[
                r#"{"start_t":0.0,"end_t":5.0}"#,
                r#"{"start_t":1.0,"end_t":2.0}"#,
            ],
            0.0,
        );
        assert_eq!(
            spans(&out),
            vec![(0.0, 5.0)],
            "the run keeps the furthest end, not the last row's"
        );
    }

    #[test]
    fn rows_out_of_order_are_taken_in_start_order() {
        let out = merged(
            &[
                r#"{"start_t":4.0,"end_t":5.0}"#,
                r#"{"start_t":0.0,"end_t":1.0}"#,
            ],
            0.0,
        );
        assert_eq!(spans(&out), vec![(0.0, 1.0), (4.0, 5.0)]);
    }

    #[test]
    fn zero_length_rows_merge_like_any_other() {
        let out = merged(
            &[
                r#"{"start_t":1.0,"end_t":1.0}"#,
                r#"{"start_t":1.5,"end_t":1.5}"#,
                r#"{"start_t":3.0,"end_t":3.0}"#,
            ],
            0.5,
        );
        assert_eq!(spans(&out), vec![(1.0, 1.5), (3.0, 3.0)]);
    }

    #[test]
    fn text_joins_with_one_space_and_other_fields_are_the_first_rows() {
        let out = merged(
            &[
                r#"{"start_t":0.0,"end_t":1.0,"text":"hello","track":"speech"}"#,
                r#"{"start_t":1.0,"end_t":2.0,"text":"there","track":"other"}"#,
            ],
            0.0,
        );
        let row: Value = serde_json::from_str(&out[0]).expect("json");
        assert_eq!(row[TEXT], "hello there");
        assert_eq!(row["track"], "speech");
    }

    #[test]
    fn a_run_that_reaches_the_end_of_the_stream_is_written_by_finish() {
        let mut node =
            RowMerge::open(&[(MAX_DISTANCE.to_string(), "1".to_string())]).expect("opens");
        let out = node.merged(vec![
            r#"{"start_t":0.0,"end_t":1.0}"#.to_string(),
            r#"{"start_t":1.5,"end_t":2.0}"#.to_string(),
        ]);
        assert!(out.is_empty(), "the run is still open");
        assert_eq!(spans(&node.finish()), vec![(0.0, 2.0)]);
    }

    #[test]
    fn a_row_carrying_no_span_rides_through_untouched() {
        let mut node =
            RowMerge::open(&[(MAX_DISTANCE.to_string(), "0".to_string())]).expect("opens");
        let out = node.merged(vec![r#"{"cues":5}"#.to_string()]);
        assert_eq!(out, vec![r#"{"cues":5}"#.to_string()]);
    }

    #[test]
    fn a_frame_keeps_its_pixels_and_carries_the_runs_that_closed() {
        let mut node =
            RowMerge::open(&[(MAX_DISTANCE.to_string(), "0".to_string())]).expect("opens");
        let frame = Frame {
            pts: 7,
            data: vec![1, 2, 3, 4].into(),
            rows: vec![
                r#"{"start_t":0.0,"end_t":1.0}"#.to_string(),
                r#"{"start_t":4.0,"end_t":5.0}"#.to_string(),
            ],
        };
        let out = node.pass(frame);
        assert_eq!(out.pts, 7);
        assert_eq!(*out.data, vec![1, 2, 3, 4], "the pixels are untouched");
        assert_eq!(spans(&out.rows), vec![(0.0, 1.0)]);
    }

    fn refuse(options: &[(&str, &str)]) -> String {
        let options: Vec<(String, String)> = options
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        match RowMerge::open(&options) {
            Ok(_) => panic!("opened, and this node must be refused"),
            Err(error) => error.to_string(),
        }
    }

    #[test]
    fn a_node_with_no_max_distance_is_refused_naming_the_option() {
        let message = refuse(&[]);
        assert!(message.contains(NODE), "got: {message}");
        assert!(message.contains("max_distance=<number>"), "got: {message}");
    }

    #[test]
    fn an_option_that_is_not_max_distance_is_refused_by_name() {
        let message = refuse(&[("pred", "{}")]);
        assert!(message.contains("'pred'"), "got: {message}");
    }

    #[test]
    fn a_max_distance_that_is_not_a_number_is_refused_showing_it() {
        let message = refuse(&[(MAX_DISTANCE, "soon")]);
        assert!(message.contains("is not a number: soon"), "got: {message}");
    }

    #[test]
    fn a_negative_max_distance_is_refused() {
        let message = refuse(&[(MAX_DISTANCE, "-1")]);
        assert!(message.contains("cannot be -1"), "got: {message}");
    }
}
