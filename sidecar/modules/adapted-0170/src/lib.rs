//! A data filter of the world before codec packages, so the host's 0.17.0
//! data-filter arm is exercised by a module actually shaped that way. The
//! interface arrived in 0.17.0 and 0.18.0 carries it unchanged, so this is
//! the one older world a data filter can be built against.
//!
//! Every message of its first data pad is written back untouched on its one
//! output, at the same time, counted in microseconds.

wit_bindgen::generate!({
    path: "../../worlds/0.17.0",
    world: "data-filter-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::data_filter::{
    DataFilterMeta, Guest, Message, Meta, PadInfo, PadKind, PadMessages, Processed, Rational,
};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;

/// Microseconds per second: the output's time base is 1/1000000.
const MICROS: i128 = 1_000_000;

thread_local! {
    /// The first data pad and its time base.
    static DATA: RefCell<Option<(u32, Rational)>> = const { RefCell::new(None) };
}

/// `pts` ticks of `time_base` in microseconds, rounded down.
fn to_micros(pts: i64, time_base: Rational) -> i64 {
    let num = i128::from(pts) * i128::from(time_base.num) * MICROS;
    num.div_euclid(i128::from(time_base.den)) as i64
}

struct Adapted0170;

impl Guest for Adapted0170 {
    fn describe() -> DataFilterMeta {
        DataFilterMeta {
            meta: Meta {
                name: "adapted_0170".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: String::new(),
                pixel_formats: vec![],
                sample_formats: vec![],
                sample_rates: vec![],
                channel_counts: vec![],
                rows_language: vec![],
            },
            outputs: vec!["json".to_string()],
            time_base: Rational {
                num: 1,
                den: MICROS as i32,
            },
        }
    }

    fn init(pads: Vec<PadInfo>, _params: String) -> Result<(), String> {
        let data = pads
            .iter()
            .enumerate()
            .find(|(_, p)| p.kind == PadKind::Data)
            .map(|(index, p)| (index as u32, p.time_base));
        DATA.with(|d| *d.borrow_mut() = data);
        Ok(())
    }

    fn process(
        input: Vec<PadMessages>,
        _now: Option<i64>,
        _last: bool,
    ) -> Result<Processed, String> {
        let data = DATA.with(|d| *d.borrow());
        let mut written = Vec::new();
        if let Some((pad, time_base)) = data {
            for carried in input.into_iter().filter(|p| p.pad == pad) {
                written.extend(carried.messages.into_iter().map(|m| Message {
                    pts: to_micros(m.pts, time_base),
                    data: m.data,
                }));
            }
        }
        Ok(Processed {
            outputs: vec![written],
            rows: vec![],
        })
    }
}

export!(Adapted0170);
