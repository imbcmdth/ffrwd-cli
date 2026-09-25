//! A packet filter that weaves rows into an h264 stream as SEI messages:
//! the small version of what an index writer does, and the fixture the rows
//! input is proven against.
//!
//! A row is `{"pts": <tick>, "note": "<text>"}`, in the stream's own time
//! base. Rows arrive whenever the host has them, which is not when their
//! packet does, so they are held until a keyframe at or past their pts goes
//! by; that keyframe leaves carrying a `user_data_unregistered` SEI NAL with
//! the notes in it, and a row saying so is emitted. `ffrwd-nal` builds that
//! NAL, frames it the way the stream frames one and splices it in before the
//! first coded slice, where a prefix SEI belongs; the packet's own bytes are
//! untouched, and its pts, dts, duration and keyframe flag are handed
//! through.
//!
//! The last row it writes says what it SAW of its rows: how many calls it
//! had, how many rows arrived on the first of them, and how many arrived at
//! all. A host that starts reading rows only once the module is open leaves
//! the first number short of the last, and where a note lands then depends
//! on which thread won rather than on what the module decided.
//!
//! It also runs a one-packet lag on every pad: each call releases the
//! packets it held from the call before and holds the ones it was just
//! given, with the last call flushing. Nothing needs the lag; it is here so
//! that a filter releasing packets on a later call than it received them is
//! what the tests drive, since that is the harder thing for a host to get
//! right than handing each call straight back.

wit_bindgen::generate!({
    path: "../../wit",
    world: "packet-filter-module",
});

use std::cell::RefCell;

use crate::ffrwd::av::types::Packet;
use exports::ffrwd::av::packet_filter::{
    Arity, CodedStream, Filtered, Guest, InputStream, Meta, PacketFilterMeta, PadPackets,
};
use ffrwd_nal::config::{framing_of, Framing};
use ffrwd_nal::Select;
use serde::{Deserialize, Serialize};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"pad":{"type":"integer"},"pts":{"type":"integer"},"notes":{"type":"integer"},"bytes":{"type":"integer"},"calls":{"type":"integer"},"rows_first_call":{"type":"integer"},"rows_total":{"type":"integer"},"args":{"type":"array","items":{"type":"string"}},"note":{"type":"string"},"woven":{"type":"boolean"}}}"#;

/// The unit's own name, 16 bytes of `uuid_iso_iec_11578`. Chosen with no
/// zero byte in it so nothing in the message can start an emulation
/// sequence: the notes beside it are ASCII for the same reason, so the NAL
/// written from it comes out of the escaper unchanged.
const UUID: [u8; 16] = [
    0x66, 0x66, 0x72, 0x77, 0x64, 0x2d, 0x73, 0x65, 0x69, 0x2d, 0x74, 0x65, 0x73, 0x74, 0x21, 0x21,
];

/// Which payloads in a stream are this module's. The AV1 `metadata_type`
/// beside the UUID is never read: `describe` names h264 as the one codec
/// this can be opened for.
const SELECT: Select = Select::new(UUID, 25);

/// One row this module reads: a note and the tick it belongs at.
///
/// `arg` is the `_arg` the host writes on every row arriving through a NAMED
/// `-rows-in`, which is how a filter reading several rows arguments tells
/// them apart. Absent for an unnamed input, and the note is then woven
/// exactly as it was written; present, and the note is prefixed with it, so
/// the bytes in the stream say which argument each one came from.
#[derive(Deserialize)]
struct NoteRow {
    pts: i64,
    note: String,
    #[serde(rename = "_arg", default)]
    arg: Option<String>,
}

impl NoteRow {
    /// What this note writes into the stream: `<arg>:<note>` for a tagged
    /// row, the note itself for an untagged one.
    fn text(&self) -> String {
        match &self.arg {
            Some(arg) => format!("{arg}:{}", self.note),
            None => self.note.clone(),
        }
    }
}

/// One row this module writes: which packet carried how many notes.
#[derive(Serialize)]
struct WovenRow {
    pad: u32,
    pts: i64,
    notes: usize,
    bytes: usize,
}

#[derive(Default)]
struct State {
    pads: usize,
    /// The packets each pad received on the call before this one, waiting to
    /// be released.
    held: Vec<Vec<Packet>>,
    /// Notes not yet woven in, oldest first.
    pending: Vec<NoteRow>,
    /// How many `process` calls this instance has had.
    calls: u64,
    /// How many rows arrived on the FIRST call, and how many over the whole
    /// run. A host that starts reading rows after the packets are already
    /// queued leaves the first number short of the second, and where a note
    /// lands then depends on which thread won.
    rows_first_call: u64,
    rows_total: u64,
    /// The distinct `_arg` values seen, in the order they first arrived.
    /// Empty where the rows came in unnamed.
    args: Vec<String>,
    /// How the stream frames its NALs, read off `init`'s extradata. None
    /// only before `init` has run.
    framing: Option<Framing>,
}

/// What the instance saw of its rows, emitted once at the end.
#[derive(Serialize)]
struct ArrivalRow {
    calls: u64,
    rows_first_call: u64,
    rows_total: u64,
    args: Vec<String>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("packet_sei takes no params, got: {other}")),
    }
}

/// The payload one SEI message carries: the module's UUID, then the notes
/// joined by newlines. `Framing::insert` wraps it in the NAL and puts that
/// where a prefix SEI goes.
fn payload(notes: &[String]) -> Vec<u8> {
    let mut bytes = UUID.to_vec();
    bytes.extend_from_slice(notes.join("\n").as_bytes());
    bytes
}

/// The packets one pad releases this call, with the notes due at or before
/// each keyframe woven into it. Rows this call consumed leave `pending`.
fn weave(
    pad: u32,
    framing: Option<Framing>,
    packets: Vec<Packet>,
    pending: &mut Vec<NoteRow>,
    rows: &mut Vec<String>,
) -> Vec<Packet> {
    let Some(framing) = framing else {
        return packets;
    };
    packets
        .into_iter()
        .map(|mut packet| {
            if !packet.keyframe || pending.is_empty() {
                return packet;
            }
            let due: Vec<NoteRow> = {
                let (due, rest): (Vec<NoteRow>, Vec<NoteRow>) = std::mem::take(pending)
                    .into_iter()
                    .partition(|r| r.pts <= packet.pts);
                *pending = rest;
                due
            };
            if due.is_empty() {
                return packet;
            }
            let texts: Vec<String> = due.iter().map(NoteRow::text).collect();
            let woven = match framing.insert(&packet.data, &payload(&texts), SELECT) {
                Ok(woven) => woven,
                // A packet whose NALs cannot be read keeps the bytes the
                // encoder gave it, and the notes wait for one that can.
                Err(_) => {
                    pending.extend(due);
                    pending.sort_by_key(|r| r.pts);
                    return packet;
                }
            };
            rows.push(
                serde_json::to_string(&WovenRow {
                    pad,
                    pts: packet.pts,
                    notes: due.len(),
                    bytes: woven.len() - packet.data.len(),
                })
                .expect("a woven row serializes"),
            );
            packet.data = woven;
            packet
        })
        .collect()
}

struct PacketSei;

impl Guest for PacketSei {
    fn describe() -> PacketFilterMeta {
        PacketFilterMeta {
            meta: Meta {
                name: "packet_sei".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
                pixel_formats: vec![],
                sample_formats: vec![],
                sample_rates: vec![],
                channel_counts: vec![],
                rows_language: vec![],
            },
            // The NAL framing this writes is h264's, so h264 is the one
            // codec it can be opened for.
            video_codecs: vec!["h264".to_string()],
            audio_codecs: vec![],
            video: Arity::One,
            audio: Arity::Zero,
            data: Arity::Zero,
            reads_rows: true,
        }
    }

    fn init(streams: Vec<InputStream>, params: String) -> Result<Vec<CodedStream>, String> {
        validate_params(&params)?;
        if streams.len() != 1 {
            return Err(format!(
                "packet_sei reads one video stream, and it was opened for {}",
                streams.len()
            ));
        }
        // Annex B out of an encoder, length-prefixed out of an MP4: the
        // extradata says which, and the crate reads it.
        let framing = framing_of(&streams[0].coded.codec, &streams[0].coded.extradata)
            .map_err(|e| format!("packet_sei cannot frame this stream: {e}"))?;
        STATE.with(|s| {
            *s.borrow_mut() = State {
                pads: streams.len(),
                held: vec![Vec::new(); streams.len()],
                framing: Some(framing),
                ..State::default()
            }
        });
        // Nothing out of band changes: the SPS and PPS the stream opened
        // with still describe every picture in it.
        Ok(streams.into_iter().map(|s| s.coded).collect())
    }

    fn set_params(params: String) -> Result<(), String> {
        validate_params(&params)
    }

    fn process(pads: Vec<PadPackets>, rows: Vec<String>, last: bool) -> Filtered {
        STATE.with(|held| {
            let mut state = held.borrow_mut();
            state.calls += 1;
            state.rows_total += rows.len() as u64;
            if state.calls == 1 {
                state.rows_first_call = rows.len() as u64;
            }
            // A row this module cannot read is dropped rather than guessed
            // at; the rows it writes say how many it used.
            for row in &rows {
                if let Ok(parsed) = serde_json::from_str::<NoteRow>(row) {
                    if let Some(arg) = &parsed.arg {
                        if !state.args.iter().any(|seen| seen == arg) {
                            state.args.push(arg.clone());
                        }
                    }
                    state.pending.push(parsed);
                }
            }
            state.pending.sort_by_key(|r| r.pts);

            let mut written = Vec::new();
            let mut out = Vec::with_capacity(state.pads);
            for (index, carried) in pads.into_iter().enumerate() {
                let releasing = std::mem::replace(&mut state.held[index], carried.packets);
                let mut pending = std::mem::take(&mut state.pending);
                let packets = weave(
                    index as u32,
                    state.framing,
                    releasing,
                    &mut pending,
                    &mut written,
                );
                state.pending = pending;
                out.push(PadPackets { packets });
            }
            if last {
                // Everything still held leaves here: a packet held past the
                // final call is a packet the container never gets.
                for (index, pad) in out.iter_mut().enumerate() {
                    let flushing = std::mem::take(&mut state.held[index]);
                    let mut pending = std::mem::take(&mut state.pending);
                    pad.packets.extend(weave(
                        index as u32,
                        state.framing,
                        flushing,
                        &mut pending,
                        &mut written,
                    ));
                    state.pending = pending;
                }
            }
            let mut trailing = Vec::new();
            if last {
                // Notes no keyframe came along to carry are reported as they
                // were written, so nothing a caller sent goes unaccounted.
                for note in std::mem::take(&mut state.pending) {
                    trailing.push(
                        serde_json::to_string(&serde_json::json!({
                            "pts": note.pts,
                            "note": note.text(),
                            "woven": false
                        }))
                        .expect("a trailing row serializes"),
                    );
                }
                trailing.push(
                    serde_json::to_string(&ArrivalRow {
                        calls: state.calls,
                        rows_first_call: state.rows_first_call,
                        rows_total: state.rows_total,
                        args: state.args.clone(),
                    })
                    .expect("an arrival row serializes"),
                );
            }
            Filtered {
                pads: out,
                rows: written,
                trailing,
            }
        })
    }
}

export!(PacketSei);
