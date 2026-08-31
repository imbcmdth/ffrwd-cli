//! ffmpeg-shaped sidecar: hosts ffrwd:av filters between NUT inputs and NUT
//! outputs. Video frames and audio packets arrive and leave inside NUT, so
//! each carries the timestamp it was made with, and the geometry - a frame
//! size and pixel format, or a sample rate, channel count and sample format -
//! comes from the stream header instead of the command line.
//!
//! One module is spelled `-m <path>`, with `-params` carrying its parameters.
//! Several are a network, configured the way ffmpeg is: `-m <name>=<path>` per
//! module, a `-filter_complex` string wiring the names together, and a `-map`
//! per output naming the label it writes. Either way the frame loop is the
//! same windowed driver over a DAG of nodes - a single module is a network of
//! one.

mod graph;
mod network;
mod rowfilter;
mod subtitles;
mod windows;

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Write};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::thread;

use anyhow::{anyhow, bail, Context, Result};
use ffrwd_wasm::nut;
use ffrwd_wasm_runtime::nn;
use ffrwd_wasm_runtime::runtime::{
    self, AudioFormat, Filter, Format, Frame, Media, StreamInfo, TimeBase, VideoFormat,
};
use serde::{Deserialize, Serialize};

use network::{Binding, Network};
use windows::Windows;

const EDGE_FORMAT: &str = "nut";
const ROWS_FORMAT: &str = "ndjson";
/// The `-f` values an output may carry, for a refusal listing them.
const OUTPUT_FORMATS: [&str; 5] = [EDGE_FORMAT, ROWS_FORMAT, "srt", "webvtt", "null"];
/// The WIT package a module targets; identifies which host generation built
/// this binary, independent of any one module's own name and version. Modules
/// built against an older world load through an adapter.
const WIT_WORLD: &str = runtime::WORLD_PACKAGE;

/// `-annotations` values: read the rows an upstream sidecar sent, and write
/// this module's rows for a downstream one.
const ANNOTATIONS_IN: &str = "in";
const ANNOTATIONS_OUT: &str = "out";

/// Byte size of one frame in `pix_fmt`, or an error naming the unsupported
/// format. `yuv420p` rejects odd width or height.
fn frame_len_for(pix_fmt: &str, width: u32, height: u32) -> Result<usize> {
    match pix_fmt {
        "rgba" => Ok((width as usize) * (height as usize) * 4),
        "yuv420p" => {
            if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
                bail!("input is yuv420p, which needs even width and height, got {width}x{height}");
            }
            Ok((width as usize) * (height as usize) * 3 / 2)
        }
        other => bail!(
            "pixel format {other}: only {} are supported",
            nut::supported_pix_fmts().join(", ")
        ),
    }
}

/// Shape of the `-stream_info` JSON file.
#[derive(Deserialize)]
struct StreamInfoFile {
    index: u32,
    kind: String,
    codec: String,
    #[serde(default)]
    duration: Option<f64>,
    #[serde(default)]
    tags: HashMap<String, String>,
}

/// What a module is told about its stream when `-stream_info` is absent: what
/// the NUT header on that input says, and nothing else.
fn stream_info_from(stream: &nut::Stream) -> StreamInfo {
    let codec = match stream.sample_fmt() {
        Some("s16") => "pcm_s16le",
        Some(_) => "pcm_f32le",
        None => "rawvideo",
    };
    StreamInfo {
        index: 0,
        kind: stream.kind().to_string(),
        codec: codec.to_string(),
        duration: None,
        tags: Vec::new(),
    }
}

fn load_stream_info(path: &str) -> Result<StreamInfo> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading -stream_info {path}"))?;
    let parsed: StreamInfoFile = serde_json::from_str(&text)
        .with_context(|| format!("parsing -stream_info {path} as JSON"))?;
    Ok(StreamInfo {
        index: parsed.index,
        kind: parsed.kind,
        codec: parsed.codec,
        duration: parsed.duration,
        tags: parsed.tags.into_iter().collect(),
    })
}

/// What the input's stream header settles about every payload on it.
fn format_from_stream(stream: &nut::Stream) -> Result<Format> {
    let time_base = TimeBase {
        num: stream.time_base.num,
        den: stream.time_base.den,
    };
    let media = match stream.media {
        nut::Media::Video { width, height, .. } => {
            let pix_fmt = stream.pix_fmt().ok_or_else(|| match stream.codec_name() {
                Some(codec) => anyhow!(
                    "input carries encoded {codec}; a frame module reads decoded video ({}), \
                     and only a packet sink consumes an encoded stream",
                    nut::supported_pix_fmts().join(", ")
                ),
                None => anyhow!(
                    "input carries codec tag {}; only {} are supported",
                    stream.fourcc_name(),
                    nut::supported_pix_fmts().join(", ")
                ),
            })?;
            Media::Video(VideoFormat {
                width,
                height,
                pix_fmt,
                frame_len: frame_len_for(pix_fmt, width, height)?,
            })
        }
        nut::Media::Audio {
            sample_rate,
            channels,
        } => {
            let sample_fmt = stream.sample_fmt().ok_or_else(|| {
                anyhow!(
                    "input carries codec tag {}; only {} are supported",
                    stream.fourcc_name(),
                    nut::supported_sample_fmts().join(", ")
                )
            })?;
            Media::Audio(AudioFormat {
                sample_rate,
                channels,
                sample_fmt,
            })
        }
    };
    Ok(Format { media, time_base })
}

/// What arrives on the wire must be what the header said: a whole frame, or a
/// whole number of samples.
fn check_frame(format: &Format, frame: &[u8], index: u64) -> Result<()> {
    match format.media {
        Media::Video(video) => {
            if frame.len() != video.frame_len {
                bail!(
                    "frame {index} carries {} bytes, not the {} a {}x{} {} frame is",
                    frame.len(),
                    video.frame_len,
                    video.width,
                    video.height,
                    video.pix_fmt
                );
            }
        }
        Media::Audio(audio) => {
            let sample_len = audio.sample_len();
            if !frame.len().is_multiple_of(sample_len) {
                bail!(
                    "packet {index} carries {} bytes, not a whole number of samples: one sample of \
                     {} across {} channel(s) is {sample_len} bytes",
                    frame.len(),
                    audio.sample_fmt,
                    audio.channels
                );
            }
        }
    }
    Ok(())
}

/// What this process hosts: one module, or a network of them.
enum Modules {
    /// `-m <path>`, its parameters as `-params`.
    Single { path: String, params: String },
    /// `-m <name>=<path>` per module, wired by a `-filter_complex` string.
    Network {
        bindings: Vec<Binding>,
        wiring: String,
    },
}

/// What one output writes: the frames, the rows as they arrive, the cue
/// rows gathered into a subtitle document, or nothing at all — the output a
/// sink module's pad takes, where the module's own effects are the product.
#[derive(Clone, Copy, PartialEq, Eq)]
enum OutputKind {
    Frames,
    Rows,
    Subtitles(subtitles::Format),
    Null,
}

impl OutputKind {
    /// The `-f` value that asks for this output.
    fn format(self) -> &'static str {
        match self {
            OutputKind::Frames => EDGE_FORMAT,
            OutputKind::Rows => ROWS_FORMAT,
            OutputKind::Subtitles(format) => format.name(),
            OutputKind::Null => "null",
        }
    }
}

/// One output file: which of the network's streams it writes, in which format.
struct OutputSpec {
    /// The `-map` label. Absent for the single-module spelling, which has one
    /// stream to write.
    target: Option<String>,
    kind: OutputKind,
    path: OutputPath,
    /// The output as it was spelled, for a refusal that names it.
    spelling: String,
}

/// Which sides of this process carry the annotation stream. Both are off for
/// an ffmpeg-facing edge, which stays single-stream NUT.
#[derive(Clone, Copy, Default)]
struct Annotations {
    input: bool,
    output: bool,
}

struct Args {
    inputs: Vec<InputPath>,
    modules: Modules,
    /// What `-stream_info` said, if it was given. Without it each input's own
    /// NUT header is what a module is told.
    stream_info: Option<StreamInfo>,
    outputs: Vec<OutputSpec>,
    annotations: Annotations,
    jobs: usize,
}

enum InputPath {
    Stdin,
    File(String),
}

enum OutputPath {
    Stdout,
    File(String),
}

/// `pipe:0` and `-` mean stdin; any other `pipe:N` is an error naming what
/// is actually supported. Anything else is opened as an ordinary path -
/// named pipes arrive looking like ordinary paths and block on open by
/// nature, which is the rendezvous.
fn resolve_input_path(raw: &str) -> Result<InputPath> {
    if raw == "-" || raw == "pipe:0" {
        return Ok(InputPath::Stdin);
    }
    if let Some(n) = raw.strip_prefix("pipe:") {
        bail!("-i pipe:{n}: only stdin (pipe:0) is supported on input");
    }
    Ok(InputPath::File(raw.to_string()))
}

/// `pipe:1` and `-` mean stdout; any other `pipe:N` is an error.
fn resolve_output_path(raw: &str) -> Result<OutputPath> {
    if raw == "-" || raw == "pipe:1" {
        return Ok(OutputPath::Stdout);
    }
    if let Some(n) = raw.strip_prefix("pipe:") {
        bail!("output pipe:{n}: only stdout (pipe:1) is supported on output");
    }
    Ok(OutputPath::File(raw.to_string()))
}

fn require_edge_format(format: Option<String>, who: &str) -> Result<()> {
    match format {
        Some(f) if f == EDGE_FORMAT => Ok(()),
        Some(f) => bail!("-f {f}: only {EDGE_FORMAT} is supported ({who})"),
        None => bail!("-f is required for {who}"),
    }
}

/// The label a `-map` names, without its brackets.
fn resolve_map_target(raw: &str) -> Result<String> {
    let label = raw
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
        .ok_or_else(|| anyhow!("-map {raw}: an output names a network label, spelled [{raw}]"))?;
    if label.is_empty() {
        bail!("-map []: an output names a network label and this one is empty");
    }
    Ok(label.to_string())
}

/// One `-m` value: `name=path` in a network, a bare path on its own.
fn resolve_binding(raw: &str) -> Result<Binding> {
    let (name, path) = raw
        .split_once('=')
        .ok_or_else(|| anyhow!("-m {raw}: a network binds a module as -m name=path"))?;
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        bail!("-m {raw}: '{name}' is not a module name; letters, digits and underscores only");
    }
    if name == rowfilter::NODE {
        bail!(
            "-m {raw}: '{name}' is the network's own node and no module is bound to it; \
             it is spelled [a]{name}=pred=<json>[b]"
        );
    }
    if path.is_empty() {
        bail!("-m {raw}: the name '{name}' is bound to no path");
    }
    Ok(Binding {
        name: name.to_string(),
        path: path.to_string(),
    })
}

/// One `-nn` value: `name=path`, the same shape and the same name rules as
/// `-m`, binding a model file to the name a module asks the host for.
fn resolve_model(raw: &str) -> Result<nn::ModelSpec> {
    let (name, path) = raw
        .split_once('=')
        .ok_or_else(|| anyhow!("-nn {raw}: a model is bound as -nn name=path"))?;
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        bail!("-nn {raw}: '{name}' is not a model name; letters, digits and underscores only");
    }
    if path.is_empty() {
        bail!("-nn {raw}: the name '{name}' is bound to no path");
    }
    Ok(nn::ModelSpec {
        name: name.to_string(),
        path: path.into(),
    })
}

/// Takes the inference options out of the argv and leaves the rest.
///
/// They are read before the argv is dispatched so that `--describe` and
/// `--invoke` reach a module that does inference, not only the frame loop.
fn take_nn_args(argv: Vec<String>) -> Result<(nn::Config, Vec<String>)> {
    let mut config = nn::Config::default();
    let mut rest = Vec::with_capacity(argv.len());
    let mut it = argv.into_iter();
    // The environment names the target for a run whose argv does not; an
    // explicit -nn-target below replaces it.
    if let Some(target) = nn::Target::from_env()? {
        config.target = target;
    }

    while let Some(arg) = it.next() {
        let mut next = |name: &str| -> Result<String> {
            it.next().ok_or_else(|| anyhow!("{name} requires a value"))
        };
        match arg.as_str() {
            "-nn" => config.models.push(resolve_model(&next("-nn")?)?),
            "-nn-runtime" => config.runtime_dir = Some(next("-nn-runtime")?.into()),
            "-nn-target" => config.target = nn::Target::parse(&next("-nn-target")?)?,
            _ => rest.push(arg),
        }
    }

    let mut seen: Vec<&str> = Vec::new();
    for spec in &config.models {
        if seen.contains(&spec.name.as_str()) {
            bail!("-nn {}: bound twice", spec.name);
        }
        seen.push(&spec.name);
    }
    Ok((config, rest))
}

/// Takes the effect grants out of the argv and leaves the rest.
///
/// `-http <module>` lets the module at that path make outbound HTTP
/// requests; `-net <module>` lets it open UDP sockets. Each is per module
/// and repeatable; a module the argv never names gets neither. Read before
/// the argv is dispatched, the way the inference options are.
fn take_grant_args(argv: Vec<String>) -> Result<Vec<String>> {
    let mut rest = Vec::with_capacity(argv.len());
    let mut it = argv.into_iter();
    while let Some(arg) = it.next() {
        let mut next = |name: &str| -> Result<String> {
            it.next().ok_or_else(|| anyhow!("{name} requires a value"))
        };
        match arg.as_str() {
            "-http" => ffrwd_wasm_runtime::runtime::grant_http(&next("-http")?)?,
            "-net" => ffrwd_wasm_runtime::runtime::grant_net(&next("-net")?)?,
            _ => rest.push(arg),
        }
    }
    Ok(rest)
}

fn parse_args(argv: Vec<String>) -> Result<Args> {
    let mut it = argv.into_iter();

    let mut format: Option<String> = None;
    let mut inputs: Vec<InputPath> = Vec::new();
    let mut modules: Vec<String> = Vec::new();
    let mut params: Option<String> = None;
    let mut wiring: Option<String> = None;
    let mut pending_map: Option<String> = None;
    let mut outputs: Vec<OutputSpec> = Vec::new();
    let mut stream_info_path: Option<String> = None;
    let mut annotations = Annotations::default();
    let mut jobs: usize = 1;

    while let Some(arg) = it.next() {
        let mut next = |name: &str| -> Result<String> {
            it.next().ok_or_else(|| anyhow!("{name} requires a value"))
        };
        match arg.as_str() {
            "-f" => format = Some(next("-f")?),
            "-pix_fmt" | "-s" | "-r" => bail!(
                "{arg} is not accepted: the NUT stream header on -i carries the frame size, \
                 pixel format and time base"
            ),
            "-i" => {
                let raw = next("-i")?;
                require_edge_format(format.take(), "input")?;
                inputs.push(resolve_input_path(&raw)?);
            }
            "-m" => modules.push(next("-m")?),
            "-filter_complex" => {
                if wiring.is_some() {
                    bail!("second -filter_complex specified: one string wires the whole network");
                }
                wiring = Some(next("-filter_complex")?);
            }
            "-map" => {
                if pending_map.is_some() {
                    bail!("two -map options in a row: each names the output that follows it");
                }
                pending_map = Some(resolve_map_target(&next("-map")?)?);
            }
            "-params" => {
                if params.is_some() {
                    bail!("second -params specified");
                }
                params = Some(next("-params")?);
            }
            "-annotations" => {
                let side = next("-annotations")?;
                let seen = match side.as_str() {
                    ANNOTATIONS_IN => std::mem::replace(&mut annotations.input, true),
                    ANNOTATIONS_OUT => std::mem::replace(&mut annotations.output, true),
                    other => bail!(
                        "-annotations {other}: only {ANNOTATIONS_IN} and {ANNOTATIONS_OUT} are supported"
                    ),
                };
                if seen {
                    bail!("second -annotations {side} specified");
                }
            }
            "-stream_info" => stream_info_path = Some(next("-stream_info")?),
            "-jobs" => {
                let raw = next("-jobs")?;
                let parsed: usize = raw
                    .parse()
                    .map_err(|_| anyhow!("-jobs expects an integer, got {raw}"))?;
                if parsed < 1 {
                    bail!("-jobs must be >= 1, got {raw}");
                }
                jobs = parsed;
            }
            "-y" => {}
            // "-" alone is the stdin/stdout shorthand, not a flag.
            other if other != "-" && other.starts_with('-') => bail!("unknown flag: {other}"),
            other => {
                let taken = format
                    .take()
                    .ok_or_else(|| anyhow!("-f is required for output"))?;
                let kind = match taken.as_str() {
                    EDGE_FORMAT => OutputKind::Frames,
                    ROWS_FORMAT => OutputKind::Rows,
                    "srt" => OutputKind::Subtitles(subtitles::Format::Srt),
                    "webvtt" => OutputKind::Subtitles(subtitles::Format::WebVtt),
                    "null" => OutputKind::Null,
                    other => bail!(
                        "-f {other}: only {} are supported for output",
                        OUTPUT_FORMATS.join(", ")
                    ),
                };
                outputs.push(OutputSpec {
                    target: pending_map.take(),
                    kind,
                    path: resolve_output_path(other)?,
                    spelling: format!("-f {taken} {other}"),
                });
            }
        }
    }

    if let Some(target) = pending_map {
        bail!("-map [{target}] names no output: an output path must follow it");
    }
    if inputs.is_empty() {
        bail!("no input specified (-i)");
    }
    if outputs.is_empty() {
        bail!("no output specified");
    }
    if annotations.output && !outputs.iter().any(|o| o.kind == OutputKind::Frames) {
        bail!(
            "-annotations {ANNOTATIONS_OUT} needs an -f {EDGE_FORMAT} output: \
             the annotation stream rides beside the frames"
        );
    }

    let modules = build_modules(modules, params, wiring, &inputs, &outputs, jobs)?;

    let stream_info = match stream_info_path {
        Some(path) => Some(load_stream_info(&path)?),
        None => None,
    };

    Ok(Args {
        inputs,
        modules,
        stream_info,
        outputs,
        annotations,
        jobs,
    })
}

/// The `-m` table and its wiring, checked against the rest of the command
/// line. A `-filter_complex` is what makes this a network; without one the
/// single-module spelling holds and every option that only a network can use
/// is refused by name.
fn build_modules(
    modules: Vec<String>,
    params: Option<String>,
    wiring: Option<String>,
    inputs: &[InputPath],
    outputs: &[OutputSpec],
    jobs: usize,
) -> Result<Modules> {
    // A network may be built entirely of nodes the host answers for, and
    // binds nothing; the wiring is then what says there is anything to run.
    if modules.is_empty() && wiring.is_none() {
        bail!("no module specified (-m)");
    }

    let Some(wiring) = wiring else {
        if modules.len() > 1 {
            bail!(
                "{} modules were given with -m and no -filter_complex wires them",
                modules.len()
            );
        }
        if inputs.len() > 1 {
            bail!("a single module reads one stream; a second -i needs a -filter_complex");
        }
        if let Some(output) = outputs.iter().find(|o| o.target.is_some()) {
            let target = output.target.as_deref().unwrap_or_default();
            bail!("-map [{target}] names a label of a network, and no -filter_complex wires one");
        }
        check_one_output_per_format(outputs)?;
        return Ok(Modules::Single {
            path: modules.into_iter().next().expect("one module"),
            params: params.unwrap_or_default(),
        });
    };

    if params.is_some() {
        bail!(
            "-params names the parameters of one module; a network writes each node's \
             parameters into -filter_complex"
        );
    }
    if jobs > 1 {
        bail!("-jobs {jobs}: a module network runs as one job, since its nodes hand frames to each other in order");
    }
    if let Some(output) = outputs.iter().find(|o| o.target.is_none()) {
        bail!(
            "the -f {} output has no -map naming which of the network's labels it writes",
            output.kind.format()
        );
    }
    let bindings = modules
        .iter()
        .map(|raw| resolve_binding(raw))
        .collect::<Result<Vec<_>>>()?;
    Ok(Modules::Network { bindings, wiring })
}

/// A single module writes one stream, so it fills one output of each format.
/// A network names a label per output and may write several of the same
/// format, one per label.
fn check_one_output_per_format(outputs: &[OutputSpec]) -> Result<()> {
    for (index, output) in outputs.iter().enumerate() {
        let format = output.kind.format();
        if outputs[..index].iter().any(|o| o.kind == output.kind) {
            bail!("second -f {format} output specified: at most one is allowed");
        }
    }
    Ok(())
}

enum InputReader {
    Stdin(io::Stdin),
    File(File),
}

impl Read for InputReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            InputReader::Stdin(r) => r.read(buf),
            InputReader::File(r) => r.read(buf),
        }
    }
}

enum OutputWriter {
    Stdout(io::Stdout),
    File(File),
}

impl Write for OutputWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            OutputWriter::Stdout(w) => w.write(buf),
            OutputWriter::File(w) => w.write(buf),
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        match self {
            OutputWriter::Stdout(w) => w.flush(),
            OutputWriter::File(w) => w.flush(),
        }
    }
}

/// The demuxed input, and the muxed frame output, as they are actually built.
type Input = nut::Demuxer<io::BufReader<InputReader>>;
type FrameOutput = nut::Muxer<io::BufWriter<OutputWriter>>;
type RowOutput = io::BufWriter<OutputWriter>;

fn open_input(path: &InputPath) -> Result<InputReader> {
    match path {
        InputPath::Stdin => Ok(InputReader::Stdin(io::stdin())),
        InputPath::File(p) => File::open(p)
            .map(InputReader::File)
            .with_context(|| format!("opening input {p}")),
    }
}

/// Outputs always create/truncate; `-y` is accepted for compatibility but
/// changes nothing.
fn open_output(path: &OutputPath) -> Result<OutputWriter> {
    match path {
        OutputPath::Stdout => Ok(OutputWriter::Stdout(io::stdout())),
        OutputPath::File(p) => File::create(p)
            .map(OutputWriter::File)
            .with_context(|| format!("opening output {p}")),
    }
}

fn open_row_output(path: &OutputPath) -> Result<RowOutput> {
    Ok(io::BufWriter::with_capacity(1 << 20, open_output(path)?))
}

/// The frame output repeats the stream header of the input its frames came
/// from, so geometry and time base reach the next process unchanged. With
/// `annotations` it also declares the annotation stream, which only another
/// sidecar reads.
fn open_frame_output(
    path: &OutputPath,
    stream: &nut::Stream,
    annotations: bool,
) -> Result<FrameOutput> {
    let writer = io::BufWriter::with_capacity(1 << 20, open_output(path)?);
    if annotations {
        nut::Muxer::with_annotations(writer, stream)
    } else {
        nut::Muxer::new(writer, stream)
    }
    .context("writing the NUT output header")
}

/// Writes one ndjson row per line.
fn write_rows<W: Write>(writer: &mut W, rows: &[String]) -> Result<()> {
    for row in rows {
        writer.write_all(row.as_bytes())?;
        writer.write_all(b"\n")?;
    }
    Ok(())
}

/// Stamps `pts` (the tick the frame carried) and `time` (seconds, through the
/// stream's time base) onto one row - unless the module's own row already
/// carries that key under that name, in which case the module's value wins: a
/// module that states its own time knows better than the host's guess, so the
/// host adds nothing under a name already there. A row that is not a JSON
/// object is left exactly as the module wrote it, since there is nothing to
/// stamp onto.
fn stamp_row(row: &str, pts: i64, time_base: nut::TimeBase) -> String {
    let Ok(serde_json::Value::Object(mut object)) = serde_json::from_str::<serde_json::Value>(row)
    else {
        return row.to_string();
    };
    object
        .entry("pts")
        .or_insert_with(|| serde_json::Value::from(pts));
    object
        .entry("time")
        .or_insert_with(|| serde_json::Value::from(time_base.seconds(pts)));
    serde_json::to_string(&object).unwrap_or_else(|_| row.to_string())
}

/// A subtitle output: where the document goes, and the cues gathered for it so
/// far. A cue's end time is its own, so the whole document is written when the
/// stream ends rather than a cue at a time.
struct SubtitleOutput {
    writer: RowOutput,
    document: subtitles::Document,
    /// The output as it was spelled, for a refusal that names it.
    spelling: String,
}

impl SubtitleOutput {
    fn push(&mut self, rows: &[String]) -> Result<()> {
        for row in rows {
            self.document.push_row(row, &self.spelling)?;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        self.writer.write_all(self.document.render().as_bytes())?;
        self.writer.flush()?;
        Ok(())
    }
}

/// Where one output's frames go, and the one thing the host checks about them
/// that no single module can: that the stream leaving this process never steps
/// backwards in time.
struct Sink {
    /// The node whose output this writes.
    node: usize,
    frames: Option<FrameOutput>,
    rows: Option<RowOutput>,
    subtitles: Option<SubtitleOutput>,
    annotations: bool,
    last_pts: Option<i64>,
    /// The stream's time base, for stamping `time` onto the rows this sink
    /// writes.
    time_base: nut::TimeBase,
}

impl Sink {
    fn open(
        output: &OutputSpec,
        node: usize,
        stream: &nut::Stream,
        annotations: bool,
    ) -> Result<Sink> {
        let mut sink = Sink {
            node,
            frames: None,
            rows: None,
            subtitles: None,
            annotations: annotations && output.kind == OutputKind::Frames,
            last_pts: None,
            time_base: stream.time_base,
        };
        match output.kind {
            OutputKind::Frames => {
                sink.frames = Some(open_frame_output(&output.path, stream, annotations)?);
            }
            OutputKind::Rows => sink.rows = Some(open_row_output(&output.path)?),
            OutputKind::Subtitles(format) => {
                sink.subtitles = Some(SubtitleOutput {
                    writer: open_row_output(&output.path)?,
                    document: subtitles::Document::new(format),
                    spelling: output.spelling.clone(),
                });
            }
            // A null output opens nothing: the monotonic-pts check still runs.
            OutputKind::Null => {}
        }
        Ok(sink)
    }

    fn write(&mut self, module: &str, frames: &[Frame]) -> Result<()> {
        for frame in frames {
            if let Some(previous) = self.last_pts {
                if frame.pts < previous {
                    bail!(
                        "{module} produced pts {} after pts {previous}; \
                         output timestamps never decrease",
                        frame.pts
                    );
                }
            }
            self.last_pts = Some(frame.pts);

            // Stamped once here, from the frame every row on this line rode,
            // so every writer below sees the same values rather than each
            // working them out for itself.
            let stamped: Vec<String> = frame
                .rows
                .iter()
                .map(|row| stamp_row(row, frame.pts, self.time_base))
                .collect();

            if let Some(w) = self.frames.as_mut() {
                // Rows first: a reader has them in hand when the frame arrives.
                if self.annotations {
                    w.write_rows(frame.pts, &stamped)?;
                }
                w.write_frame(frame.pts, &frame.data)?;
            }
            if let Some(w) = self.rows.as_mut() {
                write_rows(w, &stamped)?;
            }
            if let Some(w) = self.subtitles.as_mut() {
                w.push(&stamped)?;
            }
        }
        Ok(())
    }

    /// The rows the module had no frame to put them on. On the annotation
    /// stream they are one record after every frame's; in an ndjson output
    /// they are ordinary lines at the end. They carry no `pts` or `time`
    /// stamp: there is no frame here to take one from.
    fn write_trailing(&mut self, rows: &[String]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        if let Some(w) = self.frames.as_mut() {
            if self.annotations {
                w.write_trailing(self.last_pts.unwrap_or(0), rows)?;
            }
        }
        if let Some(w) = self.rows.as_mut() {
            write_rows(w, rows)?;
        }
        if let Some(w) = self.subtitles.as_mut() {
            w.push(rows)?;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        if let Some(w) = self.frames.as_mut() {
            w.finish()?;
        }
        if let Some(w) = self.rows.as_mut() {
            w.flush()?;
        }
        if let Some(w) = self.subtitles.as_mut() {
            w.finish()?;
        }
        Ok(())
    }
}

/// One sink per output, each bound to the node it writes and opened against
/// that node's stream header.
fn open_sinks(args: &Args, nodes: &[usize], streams: &[nut::Stream]) -> Result<Vec<Sink>> {
    args.outputs
        .iter()
        .zip(nodes)
        .zip(streams)
        .map(|((output, node), stream)| Sink::open(output, *node, stream, args.annotations.output))
        .collect()
}

/// Hands each sink whatever its own node produced in this pass.
fn write_sinks(sinks: &mut [Sink], names: &[String], produced: &[Vec<Frame>]) -> Result<()> {
    for sink in sinks.iter_mut() {
        sink.write(&names[sink.node], &produced[sink.node])?;
    }
    Ok(())
}

/// Hands each sink the trailing rows its own node ended with.
fn write_sink_trailing(sinks: &mut [Sink], trailing: &[Vec<String>]) -> Result<()> {
    for sink in sinks.iter_mut() {
        sink.write_trailing(&trailing[sink.node])?;
    }
    Ok(())
}

/// Reads every input a frame at a time, feeds the network, and writes what
/// each output's node produced. An input that has ended is left alone; the
/// loop stops when they all have.
fn run_network(
    mut net: Network,
    mut readers: Vec<Input>,
    formats: &[Format],
    mut sinks: Vec<Sink>,
    names: Vec<String>,
) -> Result<()> {
    let mut buf: Vec<u8> = Vec::new();
    let mut open = vec![true; readers.len()];
    let mut index = vec![0u64; readers.len()];

    loop {
        let mut arriving: Vec<Vec<Frame>> = (0..readers.len()).map(|_| Vec::new()).collect();
        let mut any = false;
        for (input, reader) in readers.iter_mut().enumerate() {
            if !open[input] {
                continue;
            }
            let read = reader
                .read_frame(&mut buf)
                .with_context(|| format!("reading frame {} of input {input}", index[input]))?;
            match read {
                Some(pts) => {
                    check_frame(&formats[input], &buf, index[input])?;
                    arriving[input].push(Frame {
                        pts,
                        data: std::mem::take(&mut buf),
                        rows: reader.take_rows(pts),
                    });
                    index[input] += 1;
                    any = true;
                }
                None => open[input] = false,
            }
        }
        if !any {
            break;
        }
        let produced = net.advance(arriving)?;
        write_sinks(&mut sinks, &names, &produced)?;
    }

    // The trailing record sits past the last frame, so it is in hand only now
    // that every input has been read to its end.
    let arriving_trailing: Vec<Vec<String>> =
        readers.iter_mut().map(|r| r.take_trailing()).collect();
    let drained = net.drain(&arriving_trailing)?;
    write_sinks(&mut sinks, &names, &drained.frames)?;
    write_sink_trailing(&mut sinks, &drained.trailing)?;
    for sink in sinks.iter_mut() {
        sink.finish()?;
    }
    Ok(())
}

/// One window on its way to a worker. `last` marks the final one, which
/// carries the tail the strides left over.
struct Job {
    index: u64,
    frames: Vec<Frame>,
    /// Rows that arrived with no frame to ride, which only the final window
    /// carries.
    trailing: Vec<String>,
    last: bool,
}

/// One window's output, on its way back to the writer.
struct Done {
    index: u64,
    frames: Vec<Frame>,
    trailing: Vec<String>,
}

/// `-jobs` > 1, which only the single-module spelling offers: one instance per
/// worker, windows dealt round-robin and reordered on the way out.
fn run_parallel(
    args: &Args,
    module: &str,
    params: &str,
    format: &Format,
    info: &StreamInfo,
    input: Input,
) -> Result<()> {
    let jobs = args.jobs;
    let stream = input.stream().clone();

    // Frame-parallel hosting corrupts output unless a call depends only on
    // the frames it was handed; check once, on a throwaway instance, before
    // any worker starts.
    let (shape, name) = {
        let probe = open_filter(module, params, format, info)?;
        check_reads_rows(args, &probe)?;
        if !probe.shape().pure {
            bail!(
                "{}: not pure with these parameters, -jobs {jobs} would corrupt output",
                probe.name()
            );
        }
        (probe.shape(), probe.name().to_string())
    };

    // Each worker owns one Filter instance (its own wasmtime Store) and one
    // input queue; a shared bounded queue carries results back for
    // reordering. Bounded channels give backpressure so a fast reader can't
    // outrun a slow filter and blow up memory.
    let (out_tx, out_rx): (SyncSender<Result<Done>>, Receiver<Result<Done>>) =
        sync_channel(jobs * 4);

    let mut worker_txs = Vec::with_capacity(jobs);
    let mut worker_handles = Vec::with_capacity(jobs);

    for worker_id in 0..jobs {
        let (in_tx, in_rx): (SyncSender<Job>, Receiver<Job>) = sync_channel(4);
        worker_txs.push(in_tx);

        let module = module.to_string();
        let params = params.to_string();
        let format = *format;
        let stream_info = info.clone();
        let out_tx = out_tx.clone();

        let handle = thread::spawn(move || -> Result<()> {
            let mut filter = Filter::open(&module, &format, &stream_info, &params)
                .with_context(|| format!("worker {worker_id}: opening module {module}"))?;
            for job in in_rx {
                let result = filter
                    .process_window(&job.frames, &job.trailing, job.last)
                    .map(|processed| Done {
                        index: job.index,
                        frames: processed.frames,
                        trailing: processed.trailing,
                    })
                    .with_context(|| format!("{}: processing window {}", filter.name(), job.index));
                if out_tx.send(result).is_err() {
                    break;
                }
            }
            Ok(())
        });
        worker_handles.push(handle);
    }
    drop(out_tx);

    // Reader/dispatcher runs on a separate thread so filtering and writing
    // overlap with reading the input. It cuts the windows; a worker only ever
    // sees one, which is why a pure module may be spread across instances.
    let reader_module = module.to_string();
    let reader_name = name.clone();
    let reader_format = *format;
    // Moves the only copy of the per-worker senders into the reader thread.
    // When the reader finishes (EOF or error) these drop, which is what
    // lets each worker's `for job in in_rx` loop end.
    let dispatch_txs = worker_txs;
    let reader_handle: thread::JoinHandle<Result<()>> = thread::spawn(move || {
        let mut input = input;
        let mut windows = Windows::new(shape, &reader_format);
        let mut frame_index: u64 = 0;
        let mut window_index: u64 = 0;
        let mut buf: Vec<u8> = Vec::new();
        loop {
            let Some(pts) = input
                .read_frame(&mut buf)
                .with_context(|| format!("reading frame {frame_index} for {reader_module}"))?
            else {
                break;
            };
            check_frame(&reader_format, &buf, frame_index)?;
            let frame = Frame {
                pts,
                data: std::mem::take(&mut buf),
                rows: input.take_rows(pts),
            };
            frame_index += 1;
            for window in windows.push(frame, &reader_name)? {
                let job = Job {
                    index: window_index,
                    frames: window,
                    trailing: Vec::new(),
                    last: false,
                };
                window_index += 1;
                if dispatch_txs[(job.index as usize) % dispatch_txs.len()]
                    .send(job)
                    .is_err()
                {
                    return Ok(());
                }
            }
        }
        // The final call, carrying whatever the strides left over. It reaches
        // one instance, which is what "exactly once" means for a module split
        // across workers.
        let job = Job {
            index: window_index,
            frames: windows.tail(),
            trailing: input.take_trailing(),
            last: true,
        };
        let _ = dispatch_txs[(job.index as usize) % dispatch_txs.len()].send(job);
        Ok(())
    });

    // Writer/reorder loop runs on the main thread: buffer out-of-order
    // results, flush in strict window order. Each frame's rows travel with it
    // through `pending` and are written at the same flush point, so rows from
    // window i land before rows from window i+1.
    let nodes = vec![0usize; args.outputs.len()];
    let streams = vec![stream; args.outputs.len()];
    let mut sinks = open_sinks(args, &nodes, &streams)?;
    let names = vec![name];
    let mut pending: HashMap<u64, Done> = HashMap::new();
    let mut next_index: u64 = 0;
    let mut worker_err: Option<anyhow::Error> = None;

    for result in out_rx {
        match result {
            Ok(done) => {
                pending.insert(done.index, done);
            }
            Err(e) => {
                if worker_err.is_none() {
                    worker_err = Some(e);
                }
                continue;
            }
        }
        while let Some(done) = pending.remove(&next_index) {
            write_sinks(&mut sinks, &names, &[done.frames])?;
            write_sink_trailing(&mut sinks, &[done.trailing])?;
            next_index += 1;
        }
    }
    for sink in sinks.iter_mut() {
        sink.finish()?;
    }

    let reader_result = reader_handle
        .join()
        .map_err(|_| anyhow!("reader thread panicked"))?;
    for handle in worker_handles {
        handle
            .join()
            .map_err(|_| anyhow!("worker thread panicked"))??;
    }

    if let Some(e) = worker_err {
        return Err(e);
    }
    reader_result?;

    if !pending.is_empty() {
        bail!(
            "{} window(s) never reached in-order output (highest missing index {})",
            pending.len(),
            next_index
        );
    }
    Ok(())
}

/// Opens the module once. Callers that need `-jobs` > 1 use this only to
/// probe the module's shape before spinning up workers; each worker then
/// opens its own instance, since a `Filter` is single-threaded by contract.
fn open_filter(module: &str, params: &str, format: &Format, info: &StreamInfo) -> Result<Filter> {
    Filter::open(module, format, info, params).with_context(|| format!("opening module {module}"))
}

/// What a module is told about the stream it reads: `-stream_info` when it was
/// given, and the input's own NUT header when it was not.
fn stream_info(args: &Args, stream: &nut::Stream) -> StreamInfo {
    args.stream_info
        .clone()
        .unwrap_or_else(|| stream_info_from(stream))
}

/// Reads every input's headers, then runs whatever this process hosts over
/// their frames.
fn run(args: &Args) -> Result<()> {
    let mut readers: Vec<Input> = Vec::new();
    for path in &args.inputs {
        let reader = io::BufReader::with_capacity(1 << 20, open_input(path)?);
        let demuxer = if args.annotations.input {
            nut::Demuxer::open_annotated(reader)
        } else {
            nut::Demuxer::open(reader)
        }
        .context("reading the NUT input")?;
        readers.push(demuxer);
    }

    // A packet sink takes the encoded stream itself, so it is dispatched
    // before anything asks the input for decoded frames.
    if let Modules::Single { path, params } = &args.modules {
        let is_packet_sink = ffrwd_wasm_runtime::runtime::exports_packet_sink(path)
            .with_context(|| format!("opening module {path}"))?;
        if is_packet_sink {
            let input = readers.pop().expect("one input was required");
            return run_packet_sink(args, path, params, input);
        }
    }
    if let Modules::Network { bindings, .. } = &args.modules {
        for binding in bindings {
            let is_packet_sink = ffrwd_wasm_runtime::runtime::exports_packet_sink(&binding.path)
                .with_context(|| format!("opening module {}", binding.path))?;
            if is_packet_sink {
                bail!(
                    "{}: a packet sink consumes its stream whole and joins no network; \
                     host it alone with -m {}",
                    binding.name,
                    binding.path
                );
            }
        }
    }

    let formats = readers
        .iter()
        .map(|reader| format_from_stream(reader.stream()))
        .collect::<Result<Vec<Format>>>()?;
    let streams: Vec<nut::Stream> = readers.iter().map(|r| r.stream().clone()).collect();
    let infos: Vec<StreamInfo> = streams
        .iter()
        .map(|stream| stream_info(args, stream))
        .collect();

    match &args.modules {
        Modules::Single { path, params } => {
            if args.jobs > 1 {
                let input = readers.pop().expect("one input was required");
                return run_parallel(args, path, params, &formats[0], &infos[0], input);
            }
            let filter = open_filter(path, params, &formats[0], &infos[0])?;
            check_reads_rows(args, &filter)?;
            let names = vec![filter.name().to_string()];
            let net = Network::single(filter, &formats[0]);
            let nodes = vec![0usize; args.outputs.len()];
            let sink_streams = vec![streams[0].clone(); args.outputs.len()];
            let sinks = open_sinks(args, &nodes, &sink_streams)?;
            run_network(net, readers, &formats, sinks, names)
        }
        Modules::Network { bindings, wiring } => {
            let parsed = graph::parse_network(wiring)?;
            let mapped: Vec<String> = args
                .outputs
                .iter()
                .filter_map(|o| o.target.clone())
                .collect();
            let net = Network::open(&parsed, bindings, &formats, &infos, &mapped)?;
            check_network_reads_rows(args, &net)?;

            let mut nodes = Vec::with_capacity(args.outputs.len());
            let mut sink_streams = Vec::with_capacity(args.outputs.len());
            for output in &args.outputs {
                let target = output
                    .target
                    .as_deref()
                    .expect("a network output is mapped");
                let node = net.node_for(target).ok_or_else(|| {
                    anyhow!("-map [{target}] names a label the network does not write")
                })?;
                sink_streams.push(streams[net.root(node)].clone());
                nodes.push(node);
            }
            let names = (0..parsed.len())
                .map(|node| net.name(node).to_string())
                .collect();
            let sinks = open_sinks(args, &nodes, &sink_streams)?;
            run_network(net, readers, &formats, sinks, names)
        }
    }
}

/// One encoded input through a packet sink: packets handed through untouched,
/// in decode order, rows to the row outputs. No frames leave, so the only
/// outputs are rows and null.
fn run_packet_sink(args: &Args, module: &str, params: &str, mut input: Input) -> Result<()> {
    if args.jobs > 1 {
        bail!(
            "-jobs {}: a packet sink runs as one instance, since packets reach it in decode order",
            args.jobs
        );
    }
    if args.annotations.input {
        bail!(
            "a packet sink reads encoded packets and no rows arrive with them, so -annotations \
             {ANNOTATIONS_IN} has nothing to give it"
        );
    }

    let mut row_outputs: Vec<RowOutput> = Vec::new();
    for output in &args.outputs {
        match output.kind {
            OutputKind::Rows => row_outputs.push(open_row_output(&output.path)?),
            // A null output opens nothing; the module's own effects are the
            // product.
            OutputKind::Null => {}
            _ => bail!(
                "{}: a packet sink emits rows alone; its outputs are -f {ROWS_FORMAT} and -f null",
                output.spelling
            ),
        }
    }

    let stream = input.stream().clone();
    let Some(codec) = stream.codec_name() else {
        let carried = if let Some(pix_fmt) = stream.pix_fmt() {
            format!("decoded {pix_fmt} video")
        } else if let Some(sample_fmt) = stream.sample_fmt() {
            format!("{sample_fmt} audio")
        } else {
            format!("codec tag {}", stream.fourcc_name())
        };
        bail!(
            "{module} consumes encoded packets, and this input carries {carried}; \
             put it after the encoder (ffmpeg ... -c:v <codec> -f nut)"
        );
    };
    let (width, height) = stream
        .video_geometry()
        .expect("an encoded stream this wire carries is video");
    let coded = runtime::CodedStream {
        codec: codec.to_string(),
        time_base: TimeBase {
            num: stream.time_base.num,
            den: stream.time_base.den,
        },
        width,
        height,
        extradata: stream.extradata.clone(),
    };
    let info = args.stream_info.clone().unwrap_or_else(|| StreamInfo {
        index: 0,
        kind: "video".to_string(),
        codec: codec.to_string(),
        duration: None,
        tags: Vec::new(),
    });
    let mut sink = runtime::PacketSink::open(module, &coded, &info, params)
        .with_context(|| format!("opening module {module}"))?;

    let mut buf: Vec<u8> = Vec::new();
    let mut index: u64 = 0;
    loop {
        let read = input
            .read_packet(&mut buf)
            .with_context(|| format!("reading packet {index}"))?;
        let Some(packet) = read else { break };
        let emitted = sink
            .process(
                &[runtime::Packet {
                    pts: packet.pts,
                    dts: packet.dts,
                    keyframe: packet.keyframe,
                    data: std::mem::take(&mut buf),
                }],
                false,
            )
            .with_context(|| format!("{}: processing packet {index}", sink.name()))?;
        for writer in &mut row_outputs {
            write_rows(writer, &emitted.rows)?;
        }
        index += 1;
    }

    // The final call, carrying no packets: whatever the sink held back
    // arrives as rows, and the trailing rows follow them.
    let emitted = sink
        .process(&[], true)
        .with_context(|| format!("{}: the final call", sink.name()))?;
    for writer in &mut row_outputs {
        write_rows(writer, &emitted.rows)?;
        write_rows(writer, &emitted.trailing)?;
        writer.flush()?;
    }
    Ok(())
}

/// `-annotations in` on a module that neither reads the rows nor passes them
/// on would drop every one of them silently, so it is refused instead. A
/// module that only carries them through is doing something with them, and is
/// allowed.
fn check_reads_rows(args: &Args, filter: &Filter) -> Result<()> {
    if args.annotations.input && !filter.reads_rows() && !filter.forwards_rows() {
        bail!(
            "{} does not read rows or pass them on, so -annotations {ANNOTATIONS_IN} has nothing \
             to give it",
            filter.name()
        );
    }
    Ok(())
}

/// The same refusal for a network, and one more: every module that acts on
/// rows must be reachable by the ones arriving on the wire, whether it reads
/// an input itself or sits behind modules that pass rows on. A module that
/// does not pass them on ends the path, and a reader behind it would be
/// handed nothing.
fn check_network_reads_rows(args: &Args, net: &Network) -> Result<()> {
    if !args.annotations.input {
        return Ok(());
    }
    if !net.any_module_reads_rows() {
        bail!(
            "no module of this network reads rows, so -annotations {ANNOTATIONS_IN} \
             has nothing to give it; it hosts {}",
            net.modules()
        );
    }
    if let Some((reader, blocker)) = net.reader_the_rows_cannot_reach() {
        bail!(
            "{blocker} does not pass rows on, so the rows -annotations {ANNOTATIONS_IN} \
             brings in stop there and never reach {reader}, which reads them"
        );
    }
    Ok(())
}

/// One value function's schemas, as published by `list-functions()`.
#[derive(Serialize)]
struct FunctionDescription {
    name: String,
    params_schema: serde_json::Value,
    result_schema: serde_json::Value,
}

/// One JSON object describing a module, for compile-time introspection: the
/// wit world this host was built for plus everything the module's own
/// `describe()` and `list-functions()` publish. A module that does not
/// export a frame interface carries `null` in the fields only that interface
/// can supply; one that does not export the values interface carries no
/// `functions` key at all.
#[derive(Serialize)]
struct Description {
    world: &'static str,
    name: Option<String>,
    version: Option<String>,
    params_schema: Option<serde_json::Value>,
    rows_schema: Option<serde_json::Value>,
    /// The formats of one kind, and the other kind's empty: which one is
    /// filled in is what says whether this is a video module or an audio one.
    pixel_formats: Option<Vec<String>>,
    sample_formats: Option<Vec<String>>,
    /// Sample rates and channel counts the module accepts. Empty is every one
    /// of them, which is what a module that does not care publishes.
    sample_rates: Option<Vec<u32>>,
    channel_counts: Option<Vec<u32>>,
    /// Ordered param names; the rows' language is the first of these params
    /// that is set at the call. Always present, and empty when the module
    /// declares none - which every module of a world before 0.8.0 does.
    rows_language: Vec<String>,
    /// How the host must drive a windowed module. Absent for a module
    /// exporting the older per-frame interface, which is always window 1,
    /// stride 1, one-to-one, and whose purity is not knowable until it has
    /// been opened for a stream.
    #[serde(skip_serializing_if = "Option::is_none")]
    window: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stride: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pure: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    one_to_one: Option<bool>,
    /// Whether the module ACTS on the rows arriving with its frames: declared
    /// by a windowed module, and read off the exports of a per-frame one.
    /// `null` for a module with no frame interface at all.
    reads_rows: Option<bool>,
    /// Whether upstream rows may leave on the module's own output frames.
    /// `null` for a module with no frame interface at all.
    forwards_rows: Option<bool>,
    /// The ffmpeg codec names a packet sink accepts, most preferred first,
    /// and empty for every codec. Present only for a module exporting the
    /// packet sink, which is how that export is reported - the way `window`
    /// marks a windowed module.
    #[serde(skip_serializing_if = "Option::is_none")]
    codecs: Option<Vec<String>>,
    /// How many streams the module reads at once: frames arrive one per pad,
    /// in pad order, at the same timestamp. Always present, and 1 for a module
    /// of a world before 0.9.0, for a per-frame one, and for one with no frame
    /// interface at all.
    inputs: u32,
    /// Whether the component imports `wasi:nn`, and so needs a model bound
    /// with `-nn` to run at all. Read off its imports, so it is answered
    /// without a model or an ONNX Runtime present. Always present.
    nn: bool,
    /// Whether the component imports `wasi:http`, and so needs an `-http`
    /// grant to run at all. Read off its imports. Always present.
    http: bool,
    /// Whether the component imports `wasi:sockets`, and so needs a `-net`
    /// grant to reach the network. Read off its imports. Always present.
    udp: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    functions: Vec<FunctionDescription>,
    /// The older spelling of `reads_rows`, present only when it is true.
    #[serde(skip_serializing_if = "is_false")]
    meta: bool,
}

/// `true` when `b` is `false`, for `skip_serializing_if` on a bool field that
/// should be absent rather than `false` in the common case.
fn is_false(b: &bool) -> bool {
    !*b
}

/// Parses a `meta.*_schema` string as JSON; empty means "no rows" and
/// becomes `null` rather than a parse error.
fn parse_schema(schema: &str, module_name: &str, field: &str) -> Result<serde_json::Value> {
    if schema.is_empty() {
        return Ok(serde_json::Value::Null);
    }
    serde_json::from_str(schema)
        .with_context(|| format!("{module_name}: {field} is not valid JSON"))
}

/// Compiles `module_path` far enough to call whichever of `describe()` and
/// `list-functions()` its exports support, and renders the result as one
/// JSON line. Does not init an instance for any stream or frame geometry.
fn describe_module(module_path: &str) -> Result<String> {
    let has_filter = ffrwd_wasm_runtime::runtime::exports_filter(module_path)
        .with_context(|| format!("describing {module_path}"))?;
    let has_window = ffrwd_wasm_runtime::runtime::exports_window_filter(module_path)
        .with_context(|| format!("describing {module_path}"))?;
    let has_values = ffrwd_wasm_runtime::runtime::exports_values(module_path)
        .with_context(|| format!("describing {module_path}"))?;
    let has_packet = ffrwd_wasm_runtime::runtime::exports_packet_sink(module_path)
        .with_context(|| format!("describing {module_path}"))?;

    let has_frames = has_filter || has_window;
    if !has_frames && !has_values && !has_packet {
        let exports = ffrwd_wasm_runtime::runtime::exports(module_path)?;
        if exports.is_empty() {
            bail!("{module_path} exports nothing, so no describe is possible");
        }
        bail!(
            "{module_path} exports neither a filter, a packet sink, nor value functions; \
             it exports {}",
            exports.join(", ")
        );
    }
    if has_packet && has_frames {
        bail!(
            "{module_path} exports both a packet sink and a frame interface; \
             a module is one or the other"
        );
    }

    let mut description = Description {
        world: WIT_WORLD,
        name: None,
        version: None,
        params_schema: None,
        rows_schema: None,
        pixel_formats: None,
        sample_formats: None,
        sample_rates: None,
        channel_counts: None,
        rows_language: Vec::new(),
        window: None,
        stride: None,
        pure: None,
        one_to_one: None,
        reads_rows: None,
        forwards_rows: None,
        codecs: None,
        inputs: 1,
        nn: ffrwd_wasm_runtime::runtime::imports_wasi_nn(module_path)
            .with_context(|| format!("describing {module_path}"))?,
        http: ffrwd_wasm_runtime::runtime::imports_wasi_http(module_path)
            .with_context(|| format!("describing {module_path}"))?,
        udp: ffrwd_wasm_runtime::runtime::imports_wasi_sockets(module_path)
            .with_context(|| format!("describing {module_path}"))?,
        functions: Vec::new(),
        meta: false,
    };

    if has_frames {
        let described = ffrwd_wasm_runtime::runtime::describe(module_path)
            .with_context(|| format!("describing {module_path}"))?;
        let meta = described.meta;
        description.params_schema = Some(parse_schema(
            &meta.params_schema,
            &meta.name,
            "params_schema",
        )?);
        description.rows_schema = Some(parse_schema(&meta.rows_schema, &meta.name, "rows_schema")?);
        description.pixel_formats = Some(meta.pixel_formats);
        description.sample_formats = Some(meta.sample_formats);
        description.sample_rates = Some(meta.sample_rates);
        description.channel_counts = Some(meta.channel_counts);
        description.rows_language = meta.rows_language;
        description.version = Some(meta.version);
        description.name = Some(meta.name);
        if let Some(shape) = described.shape {
            description.window = Some(shape.window);
            description.stride = Some(shape.stride);
            description.pure = Some(shape.pure);
            description.one_to_one = Some(shape.one_to_one);
        }
        description.reads_rows = Some(described.reads_rows);
        description.forwards_rows = Some(described.forwards_rows);
        description.inputs = described.inputs;
        description.meta = described.reads_rows;
    }

    if has_packet {
        let described = ffrwd_wasm_runtime::runtime::describe_packet_sink(module_path)
            .with_context(|| format!("describing {module_path}"))?;
        let meta = described.meta;
        description.params_schema = Some(parse_schema(
            &meta.params_schema,
            &meta.name,
            "params_schema",
        )?);
        description.rows_schema = Some(parse_schema(&meta.rows_schema, &meta.name, "rows_schema")?);
        description.pixel_formats = Some(meta.pixel_formats);
        description.sample_formats = Some(meta.sample_formats);
        description.sample_rates = Some(meta.sample_rates);
        description.channel_counts = Some(meta.channel_counts);
        description.rows_language = meta.rows_language;
        description.version = Some(meta.version);
        description.name = Some(meta.name);
        description.codecs = Some(described.codecs);
    }

    if has_values {
        let functions = ffrwd_wasm_runtime::runtime::list_functions(module_path)
            .with_context(|| format!("listing functions in {module_path}"))?;
        description.functions = functions
            .into_iter()
            .map(|f| {
                let params_schema = parse_schema(&f.params_schema, &f.name, "params_schema")?;
                let result_schema = parse_schema(&f.result_schema, &f.name, "result_schema")?;
                Ok(FunctionDescription {
                    name: f.name,
                    params_schema,
                    result_schema,
                })
            })
            .collect::<Result<Vec<_>>>()?;
    }

    serde_json::to_string(&description).context("serializing module description")
}

/// Calls one value function at compile time: checks the module exports the
/// values interface, checks it exports `function`, then invokes it. Errors
/// name what the module does export instead of just failing.
fn invoke_module(module_path: &str, function: &str, args_json: &str) -> Result<String> {
    let has_values = ffrwd_wasm_runtime::runtime::exports_values(module_path)
        .with_context(|| format!("invoking {function} in {module_path}"))?;
    if !has_values {
        let exports = ffrwd_wasm_runtime::runtime::exports(module_path)?;
        if exports.is_empty() {
            bail!("{module_path} exports nothing, so no value functions");
        }
        bail!(
            "{module_path} exports no value functions; it exports {}",
            exports.join(", ")
        );
    }

    let functions = ffrwd_wasm_runtime::runtime::list_functions(module_path)
        .with_context(|| format!("listing functions in {module_path}"))?;
    if !functions.iter().any(|f| f.name == function) {
        let names: Vec<&str> = functions.iter().map(|f| f.name.as_str()).collect();
        bail!(
            "{module_path} does not export function {function}; it exports {}",
            if names.is_empty() {
                "no functions".to_string()
            } else {
                names.join(", ")
            }
        );
    }

    let outcome = ffrwd_wasm_runtime::runtime::invoke(module_path, function, args_json)
        .with_context(|| format!("invoking {function} in {module_path}"))?;
    match outcome {
        Ok(result) => Ok(result),
        Err(module_err) => bail!("{module_err}"),
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();

    // Read before anything else: it names the ONNX Runtime this build demands,
    // so it has to answer on a machine that has none yet.
    if argv.first().map(String::as_str) == Some("--nn-info") {
        println!("{}", nn::info());
        std::process::exit(0);
    }

    // The egress policy is read before anything runs: a bad value is a
    // configuration error even for a run granting no network.
    if let Err(e) = ffrwd_wasm_runtime::egress::net_policy() {
        eprintln!("ffrwd-wasm: {e:#}");
        std::process::exit(2);
    }

    let (nn_config, raw_args) = match take_nn_args(argv) {
        Ok(split) => split,
        Err(e) => {
            eprintln!("ffrwd-wasm: {e:#}");
            std::process::exit(2);
        }
    };
    // Models load once, before any component is instantiated. A run binding
    // none touches no ONNX Runtime.
    if let Err(e) = nn::configure(&nn_config) {
        eprintln!("ffrwd-wasm: {e:#}");
        std::process::exit(2);
    }

    // Effect grants register before anything is instantiated, like models.
    let raw_args = match take_grant_args(raw_args) {
        Ok(rest) => rest,
        Err(e) => {
            eprintln!("ffrwd-wasm: {e:#}");
            std::process::exit(2);
        }
    };

    if raw_args.first().map(String::as_str) == Some("--describe") {
        match raw_args.get(1) {
            None => {
                eprintln!("ffrwd-wasm: --describe requires a module path");
                std::process::exit(2);
            }
            Some(module_path) => match describe_module(module_path) {
                Ok(json) => {
                    println!("{json}");
                    std::process::exit(0);
                }
                Err(e) => {
                    eprintln!("ffrwd-wasm: {e:#}");
                    std::process::exit(1);
                }
            },
        }
    }

    if raw_args.first().map(String::as_str) == Some("--invoke") {
        match (raw_args.get(1), raw_args.get(2), raw_args.get(3)) {
            (Some(module_path), Some(function), Some(args_json)) => {
                match invoke_module(module_path, function, args_json) {
                    Ok(result) => {
                        println!("{result}");
                        std::process::exit(0);
                    }
                    Err(e) => {
                        eprintln!("ffrwd-wasm: {e:#}");
                        std::process::exit(1);
                    }
                }
            }
            _ => {
                eprintln!(
                    "ffrwd-wasm: --invoke requires a module path, a function name, and args JSON"
                );
                std::process::exit(2);
            }
        }
    }

    let args = match parse_args(raw_args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("ffrwd-wasm: {e:#}");
            std::process::exit(2);
        }
    };

    if let Err(e) = run(&args) {
        eprintln!("ffrwd-wasm: {e:#}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod stamp_row_tests {
    use super::stamp_row;
    use ffrwd_wasm::nut::TimeBase;

    const TB: TimeBase = TimeBase { num: 1, den: 25 };

    fn parse(row: &str) -> serde_json::Value {
        serde_json::from_str(row).unwrap_or_else(|e| panic!("parsing {row:?}: {e}"))
    }

    #[test]
    fn a_row_with_neither_key_gets_both() {
        let stamped = stamp_row(r#"{"x":1}"#, 50, TB);
        let value = parse(&stamped);
        assert_eq!(value["x"], 1);
        assert_eq!(value["pts"], 50);
        assert_eq!(value["time"], 2.0);
    }

    #[test]
    fn a_row_that_already_names_pts_keeps_its_own_value() {
        // The module's pts (999) disagrees with the frame it rode (50), which
        // is the case that tells apart "kept" from "overwritten with the same
        // number by coincidence".
        let stamped = stamp_row(r#"{"pts":999}"#, 50, TB);
        let value = parse(&stamped);
        assert_eq!(value["pts"], 999, "the module's own pts is not clobbered");
        assert_eq!(
            value["time"], 2.0,
            "time is still the host's, since the row had none"
        );
    }

    #[test]
    fn a_row_that_already_names_time_keeps_its_own_value() {
        let stamped = stamp_row(r#"{"time":-1.0}"#, 50, TB);
        let value = parse(&stamped);
        assert_eq!(
            value["time"], -1.0,
            "the module's own time is not clobbered"
        );
        assert_eq!(
            value["pts"], 50,
            "pts is still the host's, since the row had none"
        );
    }

    #[test]
    fn a_row_that_already_names_both_is_left_alone() {
        let stamped = stamp_row(r#"{"pts":7,"time":0.5}"#, 50, TB);
        let value = parse(&stamped);
        assert_eq!(value["pts"], 7);
        assert_eq!(value["time"], 0.5);
    }

    #[test]
    fn a_row_that_is_not_a_json_object_is_returned_unchanged() {
        assert_eq!(stamp_row("not json", 50, TB), "not json");
        assert_eq!(stamp_row("[1,2,3]", 50, TB), "[1,2,3]");
    }
}
