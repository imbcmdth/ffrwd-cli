//! The node: the one model every module runs as.
//!
//! A node has typed input ports, typed output ports and a clock, and one tick
//! of the clock is one `process` call. A module built against `ffrwd:av`
//! 0.19.0 declares its own shape; a module of every older world is adapted
//! onto one by the host (`crate::adapters`), so a host drives one thing.
//!
//! The types here are the WIT's `node-types`, `node-tick` and `node` records
//! in the host's own spelling, plus the few things only an adapter needs: rows
//! riding frames and rows with no frame to ride, which the old worlds carried
//! and the node world does not.

use std::sync::Arc;

use anyhow::{bail, Result};

use crate::runtime::{
    AudioFormat, CodedStream, Message, Packet, RenditionMeta, StreamInfo, TimeBase, VideoFormat,
    Wants,
};

/// What travels on a port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortKind {
    Video,
    Audio,
    Data,
    Packets,
}

impl PortKind {
    pub fn name(self) -> &'static str {
        match self {
            PortKind::Video => "video",
            PortKind::Audio => "audio",
            PortKind::Data => "data",
            PortKind::Packets => "packets",
        }
    }

    /// Video and audio carry frames; data and packets carry messages.
    pub fn is_frames(self) -> bool {
        matches!(self, PortKind::Video | PortKind::Audio)
    }
}

/// What a node does with the rows arriving on an input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowsUse {
    Ignore,
    PerFrame,
    State,
}

/// How a hold input's offset between source time and clock time is fixed.
#[derive(Debug, Clone, PartialEq)]
pub enum Anchor {
    SharedClock,
    FirstFrame,
    Tagged(String),
}

/// A frame input paired by time: the newest frame at or before the tick.
#[derive(Debug, Clone, PartialEq)]
pub struct Hold {
    pub anchor: Anchor,
    pub lead: f64,
    pub linger: Option<f64>,
    pub timeout: Option<f64>,
    pub group: Option<String>,
    pub port_param: Option<String>,
}

/// A message input paired by time: every message in the tick's interval,
/// extended by `ahead`.
#[derive(Debug, Clone, PartialEq)]
pub struct Interval {
    pub latency: Option<f64>,
    pub ahead: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pairing {
    Lockstep,
    Hold(Hold),
    Interval(Interval),
    Arrival,
    /// An adapted data filter's data pads: every message at or before the
    /// tick's time, held while it is ahead of the clock, and a pad read from
    /// a regular file read past the tick before the tick is made. The node
    /// world's interval pairing hands a tick what starts at its time
    /// instead, so this lives only in the adapter for the 0.17 world.
    AtOrBefore,
}

/// A ratio of two whole numbers, as the WIT spells one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rational {
    pub num: i32,
    pub den: i32,
}

impl Rational {
    /// The time base this ratio is, refusing one with no positive
    /// denominator or a negative numerator.
    pub fn time_base(self, what: &str) -> Result<TimeBase> {
        if self.num < 0 || self.den <= 0 {
            bail!("{what}: {}/{} is not a time base", self.num, self.den);
        }
        Ok(TimeBase {
            num: self.num as u64,
            den: self.den as u64,
        })
    }
}

/// What an input accepts. Empty lists accept anything of the kind.
#[derive(Debug, Clone, PartialEq)]
pub struct Accepts {
    pub pixel_formats: Vec<String>,
    pub sample_formats: Vec<String>,
    pub sample_rates: Vec<u32>,
    pub channel_counts: Vec<u32>,
    pub codecs: Vec<String>,
    pub wants: Wants,
    pub like: Option<String>,
}

impl Default for Accepts {
    fn default() -> Self {
        Accepts {
            pixel_formats: Vec::new(),
            sample_formats: Vec::new(),
            sample_rates: Vec::new(),
            channel_counts: Vec::new(),
            codecs: Vec::new(),
            wants: Wants::All,
            like: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct InputPort {
    pub name: String,
    pub kind: PortKind,
    pub required: bool,
    pub many: bool,
    pub pairing: Pairing,
    pub rows: RowsUse,
    pub window: u32,
    pub stride: u32,
    pub accepts: Accepts,
    pub schema: Option<String>,
}

/// A colorimetry a module declares, in ffmpeg's own names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColorSpec {
    pub range: String,
    pub primaries: String,
    pub trc: String,
    pub space: String,
}

/// A video output's frames as a module declares them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoSpec {
    pub width: u32,
    pub height: u32,
    pub pix_fmt: String,
    pub color: Option<ColorSpec>,
}

/// An audio output's samples as a module declares them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioSpec {
    pub sample_rate: u32,
    pub channels: u32,
    pub sample_fmt: String,
    pub channel_layout: Option<String>,
}

/// An input's format with one field overridden.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LikeInput {
    pub port: String,
    pub pixel_format: Option<String>,
    pub sample_format: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OutputFormat {
    Video(VideoSpec),
    Audio(AudioSpec),
    Data(String),
    Packets(CodedStream),
    Like(LikeInput),
}

#[derive(Debug, Clone, PartialEq)]
pub struct OutputPort {
    pub name: String,
    pub kind: PortKind,
    pub format: Option<OutputFormat>,
    pub time_base: Option<Rational>,
    pub latency: f64,
    pub schema: Option<String>,
    pub row: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Clock {
    Input(String),
    Rate(Rational),
    RateOf(String),
    SelfClocked,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodeShape {
    pub inputs: Vec<InputPort>,
    pub outputs: Vec<OutputPort>,
    pub clock: Clock,
    pub pure: bool,
    pub one_to_one: bool,
    pub bounded: bool,
    pub relation: Vec<String>,
}

impl NodeShape {
    pub fn input(&self, name: &str) -> Option<&InputPort> {
        self.inputs.iter().find(|p| p.name == name)
    }

    pub fn output_index(&self, name: &str) -> Option<usize> {
        self.outputs.iter().position(|p| p.name == name)
    }

    /// The clock input's port, for a node clocked by one.
    pub fn clock_input(&self) -> Option<&InputPort> {
        match &self.clock {
            Clock::Input(name) => self.input(name),
            _ => None,
        }
    }
}

/// The format a bound stream arrives in, resolved.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamFormat {
    Video(VideoFormat),
    Audio(AudioFormat),
    Data(String),
    Packets(CodedStream),
}

impl StreamFormat {
    pub fn kind(&self) -> PortKind {
        match self {
            StreamFormat::Video(_) => PortKind::Video,
            StreamFormat::Audio(_) => PortKind::Audio,
            StreamFormat::Data(_) => PortKind::Data,
            StreamFormat::Packets(_) => PortKind::Packets,
        }
    }
}

/// One stream bound to an input port at `init`.
#[derive(Debug, Clone)]
pub struct BoundStream {
    pub port: String,
    pub id: u32,
    pub info: StreamInfo,
    pub time_base: TimeBase,
    pub format: StreamFormat,
    pub rendition: RenditionMeta,
    pub row: Option<u32>,
    pub decode_delay: u32,
    pub latency: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimedRows {
    pub pts: i64,
    pub rows: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeedStart {
    pub tags: Vec<(String, String)>,
    pub first_pts: i64,
    pub at: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Feed {
    pub start: FeedStart,
    pub ends: Option<i64>,
}

/// One frame as a tick holds it: a picture, or a run of samples.
#[derive(Debug, Clone)]
pub struct TickFrame {
    pub pts: i64,
    pub duration: Option<i64>,
    pub data: Arc<Vec<u8>>,
    pub rows: Vec<String>,
}

impl From<crate::runtime::Frame> for TickFrame {
    fn from(frame: crate::runtime::Frame) -> Self {
        TickFrame {
            pts: frame.pts,
            duration: None,
            data: frame.data,
            rows: frame.rows,
        }
    }
}

/// One bound stream's share of a tick.
#[derive(Debug, Clone, Default)]
pub struct TickStream {
    pub id: u32,
    pub frames: Vec<TickFrame>,
    pub messages: Vec<Message>,
    pub packets: Vec<Packet>,
    pub earlier_rows: Vec<TimedRows>,
    pub feed: Option<Feed>,
    /// The stream as the host knows it this tick, where that differs from
    /// what `init` was told: a hold input's current source, with its time
    /// base.
    pub info: Option<(StreamInfo, TimeBase)>,
    /// How far the producer has said it is done, in the stream's time base:
    /// the newest progress mark that arrived. Never handed to a module; an
    /// adapter hands it on.
    pub progress: Option<i64>,
    /// Old worlds only: rows that arrived with no frame to ride.
    pub trailing: Vec<String>,
}

/// What one call is handed.
#[derive(Debug, Clone)]
pub struct Tick {
    pub pts: i64,
    pub time_base: TimeBase,
    pub last: bool,
    pub streams: Vec<TickStream>,
}

impl Tick {
    pub fn stream(&self, id: u32) -> Option<&TickStream> {
        self.streams.iter().find(|s| s.id == id)
    }
}

/// A frame leaving: new bytes, or an input's handed back by `same`, which
/// the host has already resolved to the input's own buffer.
#[derive(Debug, Clone)]
pub struct OutFrame {
    pub pts: i64,
    pub duration: Option<i64>,
    pub data: Arc<Vec<u8>>,
    /// Old worlds only: the rows riding this frame.
    pub rows: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum Payload {
    Frame(OutFrame),
    /// An empty message is a progress mark.
    Message(Message),
    Packet(Packet),
    /// Old worlds only: rows leaving on the port with no frame to ride.
    Rows(Vec<String>),
}

#[derive(Debug, Clone)]
pub struct Emission {
    /// Index into the shape's outputs.
    pub port: usize,
    pub payload: Payload,
}

/// What one tick produced.
#[derive(Debug, Clone, Default)]
pub struct Emitted {
    pub items: Vec<Emission>,
    pub rows: Vec<String>,
    pub finished: bool,
}

/// One instance of a node, whatever world its module was built against.
pub trait Node: Send {
    fn name(&self) -> &str;

    fn shape(&self) -> &NodeShape;

    /// An output's format where the node itself settles it at `init`; None
    /// leaves it to the host, which reads it off the shape and the bound
    /// streams.
    fn settled_format(&self, _port: usize) -> Option<StreamFormat> {
        None
    }

    /// The decode delay a packets output's header declares, where the node
    /// settles it: the coded stream a port carries has no field for it.
    fn decode_delay(&self, _port: usize) -> Option<u32> {
        None
    }

    /// Whether the host sends the node's progress down its data edges after
    /// each tick. An adapter of an old world writes its own, the way that
    /// world's host did.
    fn host_progress(&self) -> bool {
        true
    }

    fn set_params(&mut self, params: &str) -> Result<()>;

    fn process(&mut self, tick: Tick) -> Result<Emitted>;
}

/// The refusals `shape`'s doc lists, naming the module and the port.
/// `bound` is the inputs the call binds, by name.
pub fn check_shape(shape: &NodeShape, bound: &[String], name: &str) -> Result<()> {
    for (index, port) in shape.inputs.iter().enumerate() {
        if shape.inputs[..index].iter().any(|p| p.name == port.name) {
            bail!("{name} declares input '{}' twice", port.name);
        }
    }
    for (index, port) in shape.outputs.iter().enumerate() {
        if shape.outputs[..index].iter().any(|p| p.name == port.name) {
            bail!("{name} declares output '{}' twice", port.name);
        }
    }
    for wanted in bound {
        if shape.input(wanted).is_none() {
            bail!("{name} is bound input '{wanted}', which its shape does not declare");
        }
    }

    let input_clock = match &shape.clock {
        Clock::Input(port) => {
            let Some(clock) = shape.input(port) else {
                bail!("{name} is clocked by input '{port}', which it does not declare");
            };
            if !clock.required || clock.many || clock.pairing != Pairing::Lockstep {
                bail!(
                    "{name} is clocked by input '{port}', and a clock input is required, single \
                     and lockstep"
                );
            }
            true
        }
        Clock::Rate(rate) => {
            if rate.num <= 0 || rate.den <= 0 {
                bail!(
                    "{name} ticks at {}/{}, which is no rate",
                    rate.num,
                    rate.den
                );
            }
            false
        }
        Clock::RateOf(port) => {
            if shape.input(port).is_none() {
                bail!("{name} ticks at the rate of input '{port}', which it does not declare");
            }
            false
        }
        Clock::SelfClocked => false,
    };
    let self_clocked = shape.clock == Clock::SelfClocked;

    for port in &shape.inputs {
        let at = format!("{name} input '{}'", port.name);
        match &port.pairing {
            Pairing::Lockstep if !input_clock => {
                bail!("{at} is lockstep, and {name} has no input clock to step with")
            }
            Pairing::Hold(_) if !port.kind.is_frames() => {
                bail!(
                    "{at} is {} and paired by hold, which pairs frames",
                    port.kind.name()
                )
            }
            Pairing::Interval(_) if port.kind.is_frames() => bail!(
                "{at} is {} and paired by interval, which pairs messages",
                port.kind.name()
            ),
            Pairing::Lockstep | Pairing::Hold(_) | Pairing::Interval(_) if self_clocked => {
                bail!("{at} is paired by time, and a self-clocked node takes its inputs as they arrive")
            }
            _ => {}
        }
        if port.kind == PortKind::Data && port.rows == RowsUse::Ignore {
            bail!("{at} is data and ignores its rows; a data input's rows are its messages");
        }
        if port.stride == 0 || port.stride > port.window {
            bail!(
                "{at} declares window {} and stride {}; a stride is at least 1 and at most its \
                 window",
                port.window,
                port.stride
            );
        }
        let is_clock = matches!(&shape.clock, Clock::Input(c) if *c == port.name);
        if !is_clock && (port.window != 1 || port.stride != 1) {
            bail!(
                "{at} declares window {} and stride {}; only the clock input has a window, and \
                 every other hands the tick's interval at 1/1",
                port.window,
                port.stride
            );
        }
        if let Some(like) = &port.accepts.like {
            check_like(shape, bound, like, &at)?;
        }
    }
    for port in &shape.outputs {
        if let Some(OutputFormat::Like(like)) = &port.format {
            let at = format!("{name} output '{}'", port.name);
            check_like(shape, bound, &like.port, &at)?;
        }
    }
    Ok(())
}

/// A `like` names an input that is single and bound, and a frame kind.
fn check_like(shape: &NodeShape, bound: &[String], like: &str, at: &str) -> Result<()> {
    let Some(input) = shape.input(like) else {
        bail!("{at} follows input '{like}', which is not declared");
    };
    if input.many {
        bail!("{at} follows input '{like}', which takes many streams");
    }
    if !bound.iter().any(|b| b == like) {
        bail!("{at} follows input '{like}', which this call leaves unbound");
    }
    if !input.kind.is_frames() {
        bail!(
            "{at} follows input '{like}', which carries {}",
            input.kind.name()
        );
    }
    Ok(())
}

/// The time a data port's progress stands at after a tick: the end of the
/// tick's interval less the port's latency, never below the last message's
/// pts. Both are in the port's own time base; `latency` is in seconds.
pub fn progress(interval_end: i64, latency: f64, base: TimeBase, last: Option<i64>) -> i64 {
    let held = (latency * base.den as f64 / base.num.max(1) as f64).ceil() as i64;
    let at = interval_end.saturating_sub(held);
    last.map_or(at, |last| at.max(last))
}

/// `pts` in `from` restated in `to`, rounded down.
pub fn rescale(pts: i64, from: TimeBase, to: TimeBase) -> i64 {
    let num = i128::from(pts) * i128::from(from.num) * i128::from(to.den);
    let den = i128::from(from.den) * i128::from(to.num);
    num.div_euclid(den.max(1)) as i64
}

/// A bound stream against what its port accepts; an empty list accepts
/// anything of the kind.
pub fn check_accepts(port: &InputPort, stream: &BoundStream, name: &str) -> Result<()> {
    let accepts = &port.accepts;
    let refused = |what: &str, wanted: String, got: String| {
        anyhow::anyhow!(
            "{name} input '{}' accepts {what} {wanted}, and is bound a stream of {got}; convert \
             it before it reaches {name}",
            port.name
        )
    };
    match &stream.format {
        StreamFormat::Video(v) => {
            if !accepts.pixel_formats.is_empty()
                && !accepts.pixel_formats.iter().any(|f| f == v.pix_fmt)
            {
                return Err(refused(
                    "the pixel formats",
                    accepts.pixel_formats.join(", "),
                    v.pix_fmt.to_string(),
                ));
            }
        }
        StreamFormat::Audio(a) => {
            if !accepts.sample_formats.is_empty()
                && !accepts.sample_formats.iter().any(|f| f == a.sample_fmt)
            {
                return Err(refused(
                    "the sample formats",
                    accepts.sample_formats.join(", "),
                    a.sample_fmt.to_string(),
                ));
            }
            if !accepts.sample_rates.is_empty() && !accepts.sample_rates.contains(&a.sample_rate) {
                return Err(refused(
                    "the sample rates",
                    numbers(&accepts.sample_rates),
                    format!("{} Hz", a.sample_rate),
                ));
            }
            if !accepts.channel_counts.is_empty() && !accepts.channel_counts.contains(&a.channels) {
                return Err(refused(
                    "the channel counts",
                    numbers(&accepts.channel_counts),
                    format!("{} channels", a.channels),
                ));
            }
        }
        StreamFormat::Packets(coded) => {
            if !accepts.codecs.is_empty() && !accepts.codecs.contains(&coded.codec) {
                return Err(refused(
                    "the codecs",
                    accepts.codecs.join(", "),
                    coded.codec.clone(),
                ));
            }
        }
        StreamFormat::Data(_) => {}
    }
    Ok(())
}

fn numbers(values: &[u32]) -> String {
    values
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn port(name: &str, kind: PortKind, pairing: Pairing) -> InputPort {
        InputPort {
            name: name.to_string(),
            kind,
            required: true,
            many: false,
            pairing,
            rows: if kind == PortKind::Data {
                RowsUse::PerFrame
            } else {
                RowsUse::Ignore
            },
            window: 1,
            stride: 1,
            accepts: Accepts::default(),
            schema: None,
        }
    }

    fn shape(inputs: Vec<InputPort>, clock: Clock) -> NodeShape {
        NodeShape {
            inputs,
            outputs: Vec::new(),
            clock,
            pure: true,
            one_to_one: false,
            bounded: true,
            relation: Vec::new(),
        }
    }

    fn refusal(shape: &NodeShape, bound: &[&str]) -> String {
        let bound: Vec<String> = bound.iter().map(|b| b.to_string()).collect();
        check_shape(shape, &bound, "m")
            .expect_err("refused")
            .to_string()
    }

    #[test]
    fn the_clock_input_is_required_single_and_lockstep() {
        let mut v = port("v", PortKind::Video, Pairing::Lockstep);
        v.required = false;
        let message = refusal(&shape(vec![v], Clock::Input("v".into())), &["v"]);
        assert!(
            message.contains("required, single and lockstep"),
            "{message}"
        );
        let message = refusal(&shape(vec![], Clock::Input("v".into())), &[]);
        assert!(message.contains("does not declare"), "{message}");
    }

    #[test]
    fn lockstep_needs_an_input_clock_and_a_self_clocked_node_takes_arrivals() {
        let rate = Clock::Rate(Rational { num: 25, den: 1 });
        let message = refusal(
            &shape(vec![port("v", PortKind::Video, Pairing::Lockstep)], rate),
            &["v"],
        );
        assert!(message.contains("no input clock"), "{message}");
        let interval = Pairing::Interval(Interval {
            latency: None,
            ahead: 0.0,
        });
        let message = refusal(
            &shape(
                vec![port("d", PortKind::Data, interval)],
                Clock::SelfClocked,
            ),
            &["d"],
        );
        assert!(message.contains("as they arrive"), "{message}");
    }

    #[test]
    fn hold_pairs_frames_and_interval_pairs_messages() {
        let hold = Pairing::Hold(Hold {
            anchor: Anchor::SharedClock,
            lead: 0.0,
            linger: None,
            timeout: None,
            group: None,
            port_param: None,
        });
        let clock = || Clock::Rate(Rational { num: 25, den: 1 });
        let message = refusal(&shape(vec![port("d", PortKind::Data, hold)], clock()), &[]);
        assert!(message.contains("pairs frames"), "{message}");
        let interval = Pairing::Interval(Interval {
            latency: None,
            ahead: 0.0,
        });
        let message = refusal(
            &shape(vec![port("v", PortKind::Video, interval)], clock()),
            &[],
        );
        assert!(message.contains("pairs messages"), "{message}");
    }

    #[test]
    fn a_data_input_reads_its_rows_and_a_stride_stays_inside_its_window() {
        let mut d = port("d", PortKind::Data, Pairing::Lockstep);
        d.rows = RowsUse::Ignore;
        let message = refusal(&shape(vec![d], Clock::Input("d".into())), &["d"]);
        assert!(message.contains("ignores its rows"), "{message}");
        let mut v = port("v", PortKind::Video, Pairing::Lockstep);
        v.window = 2;
        v.stride = 3;
        let message = refusal(&shape(vec![v], Clock::Input("v".into())), &["v"]);
        assert!(message.contains("at most its window"), "{message}");
    }

    #[test]
    fn like_names_a_single_bound_frame_input() {
        let mut s = shape(
            vec![port("v", PortKind::Video, Pairing::Lockstep)],
            Clock::Input("v".into()),
        );
        s.outputs.push(OutputPort {
            name: "out".into(),
            kind: PortKind::Video,
            format: Some(OutputFormat::Like(LikeInput {
                port: "w".into(),
                pixel_format: None,
                sample_format: None,
            })),
            time_base: None,
            latency: 0.0,
            schema: None,
            row: None,
        });
        let message = refusal(&s, &["v"]);
        assert!(message.contains("not declared"), "{message}");
        if let Some(OutputFormat::Like(like)) = &mut s.outputs[0].format {
            like.port = "v".into();
        }
        assert!(check_shape(&s, &["v".to_string()], "m").is_ok());
    }

    #[test]
    fn progress_is_the_intervals_end_less_the_latency_and_never_behind_a_message() {
        let micros = TimeBase {
            num: 1,
            den: 1_000_000,
        };
        assert_eq!(progress(2_000_000, 0.5, micros, None), 1_500_000);
        assert_eq!(progress(2_000_000, 0.5, micros, Some(1_800_000)), 1_800_000);
        assert_eq!(progress(2_000_000, 0.0, micros, Some(10)), 2_000_000);
    }
}
