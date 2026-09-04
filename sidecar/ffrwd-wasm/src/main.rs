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
mod rows_chain;
mod scheduler;
mod subtitles;
mod windows;

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, Read, Write};
use std::sync::{Arc, Condvar, Mutex};

use anyhow::{anyhow, bail, Context, Result};
use ffrwd_wasm::nut;
use ffrwd_wasm_runtime::nn;
use ffrwd_wasm_runtime::runtime::{
    self, AudioFormat, Filter, Format, Frame, Media, StreamInfo, TimeBase, VideoFormat,
};
use serde::{Deserialize, Serialize};

use network::{Binding, Network};

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
                color: color_from(&stream.media),
            })
        }
        nut::Media::Audio {
            sample_rate,
            channels,
        } => {
            let sample_fmt = stream
                .sample_fmt()
                .ok_or_else(|| match stream.codec_name() {
                    Some(codec) => anyhow!(
                        "input carries encoded {codec}; a frame module reads decoded audio \
                         ({}), and only a packet sink consumes an encoded stream",
                        nut::supported_sample_fmts().join(", ")
                    ),
                    None => anyhow!(
                        "input carries codec tag {}; only {} are supported",
                        stream.fourcc_name(),
                        nut::supported_sample_fmts().join(", ")
                    ),
                })?;
            Media::Audio(AudioFormat {
                sample_rate,
                channels,
                sample_fmt,
                // The NUT audio header carries a rate and a count and no
                // layout, so there is none to hand on.
                channel_layout: None,
            })
        }
    };
    Ok(Format { media, time_base })
}

/// The colorimetry a NUT video header declares, in ffmpeg's own names. The
/// wire codes the range and the matrix together (1 and 2 are Rec 601 and
/// Rec 709 in the tv range; 16 on top of either is the pc range) and says
/// nothing of primaries or transfer, which stay "unknown". Rec 601 names
/// its matrix without choosing 525 or 625 lines - the coefficients are the
/// same - and bt470bg is the name used here.
fn color_from(media: &nut::Media) -> Option<runtime::ColorInfo> {
    let nut::Media::Video {
        colorspace_type, ..
    } = media
    else {
        return None;
    };
    let space = match colorspace_type & !16 {
        1 => "bt470bg",
        2 => "bt709",
        _ => return None,
    };
    Some(runtime::ColorInfo {
        range: if colorspace_type & 16 != 0 {
            "pc"
        } else {
            "tv"
        },
        primaries: "unknown",
        trc: "unknown",
        space,
    })
}

/// The pixel aspect ratio a NUT video header declares. The wire spells
/// unknown as 0/0, and this wire's own headers write 1/1.
fn aspect_from(sample_width: u64, sample_height: u64) -> Option<(i32, i32)> {
    let num = i32::try_from(sample_width).ok()?;
    let den = i32::try_from(sample_height).ok()?;
    (num > 0 && den > 0).then_some((num, den))
}

/// The h264 profile and level: bytes 1 and 3 of the SPS payload, the same
/// two an avcC copies into its own header. The NUT wire carries Annex-B
/// extradata, so the SPS is found behind its start code; extradata already
/// shaped as an avcC (first byte 1, a version no Annex-B stream starts
/// with) reads the same bytes off its header instead.
fn h264_profile_level(extradata: &[u8]) -> Option<(i32, i32)> {
    if extradata.first() == Some(&1) {
        return match extradata {
            [_, profile, _, level, ..] => Some((i32::from(*profile), i32::from(*level))),
            _ => None,
        };
    }
    let mut at = 0usize;
    while at + 4 < extradata.len() {
        if extradata[at..at + 3] != [0, 0, 1] {
            at += 1;
            continue;
        }
        let nal = &extradata[at + 3..];
        if nal[0] & 0x1f == 7 {
            return match nal {
                [_, profile, _, level, ..] => Some((i32::from(*profile), i32::from(*level))),
                _ => None,
            };
        }
        at += 3;
    }
    None
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

/// One `-m <path> -rows-from <index>` pair: a rows module hosted in this
/// process, carrying no stream of its own. `slot` is its own position among
/// the line's `-m` flags and `source` what its `-rows-from` named, checked as
/// the line is parsed to be an earlier `-m`: the two are what an output's
/// `-rows` walks back through to the chain its rows flow along.
struct RowsModuleSpec {
    path: String,
    slot: usize,
    source: usize,
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
    /// `-rows`: whose rows this document holds, as a position among the
    /// line's `-m` flags - a module's own rows, or a rows module's output.
    /// Absent for an output that carries frames, and for the one rows
    /// document a line ordinarily writes.
    rows: Option<usize>,
    /// The output as it was spelled, for a refusal that names it.
    spelling: String,
}

impl OutputSpec {
    /// Whether this output writes rows rather than frames.
    fn rows_bearing(&self) -> bool {
        matches!(self.kind, OutputKind::Rows | OutputKind::Subtitles(_))
    }
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
    /// One slot per input, in the same order: what the `-pad` that followed
    /// its `-i` said, or None where there was no `-pad`.
    pads: Vec<Option<PadSpec>>,
    modules: Modules,
    /// What `-stream_info` said, if it was given. Without it each input's own
    /// NUT header is what a module is told.
    stream_info: Option<StreamInfo>,
    outputs: Vec<OutputSpec>,
    annotations: Annotations,
    /// `-rows-in`: where a rows module's input rows come from. A rows
    /// module reads no `-i` stream at all, so this is its only input.
    rows_in: Option<InputPath>,
    /// Every `-m <path> -rows-from <index>` on the line, in the order they
    /// were given - the hops a rows-bearing output's rows may flow through
    /// before they are written.
    rows_chain: Vec<RowsModuleSpec>,
    /// The `-jobs` cap on worker threads, if one was given. The pool is the
    /// machine's effective core count either way; the cap only lowers it,
    /// and `-jobs 1` is the serial escape hatch.
    jobs: Option<usize>,
}

/// `-pad`'s JSON, following one packet sink `-i`: which relation row this
/// pad belongs to, and what the source read of it. Absent fields stay
/// None - the same "nothing said" a pad with no `-pad` at all gets: row is
/// its own index among the sink's inputs, and every rendition field is
/// None.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
struct PadSpec {
    row: Option<u32>,
    #[serde(default)]
    rendition: PadRendition,
}

/// `-pad`'s `rendition` object: a row's name, bitrate and codec string,
/// exactly as the manifest or catalog said them. Mirrors
/// `runtime::RenditionMeta`.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
struct PadRendition {
    name: Option<String>,
    bandwidth: Option<u64>,
    codecs: Option<String>,
    language: Option<String>,
}

#[derive(Clone)]
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
/// `-nn-exclude` is repeatable and deduplicated; it drops a provider from a
/// `-nn-target gpu` walk for the whole process.
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
            "-nn-exclude" => {
                let provider = nn::parse_exclude(&next("-nn-exclude")?)?;
                if !config.exclude.contains(&provider) {
                    config.exclude.push(provider);
                }
            }
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
    let mut pads: Vec<Option<PadSpec>> = Vec::new();
    let mut modules: Vec<String> = Vec::new();
    // One slot per `-m`, in the same order: what its own `-rows-from` said,
    // or None where there was none.
    let mut rows_from: Vec<Option<usize>> = Vec::new();
    let mut params: Option<String> = None;
    let mut wiring: Option<String> = None;
    let mut pending_map: Option<String> = None;
    let mut outputs: Vec<OutputSpec> = Vec::new();
    let mut stream_info_path: Option<String> = None;
    let mut annotations = Annotations::default();
    let mut rows_in: Option<InputPath> = None;
    let mut jobs: Option<usize> = None;
    // What the `-rows` before the next output said, if one was given.
    let mut pending_rows: Option<usize> = None;

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
                pads.push(None);
            }
            "-pad" => {
                let raw = next("-pad")?;
                let spec: PadSpec =
                    serde_json::from_str(&raw).map_err(|e| anyhow!("-pad {raw}: {e}"))?;
                match pads.last_mut() {
                    Some(slot @ None) => *slot = Some(spec),
                    Some(Some(_)) => bail!("second -pad for the same -i"),
                    None => bail!("-pad with no -i before it: -pad follows the -i it names"),
                }
            }
            "-m" => {
                modules.push(next("-m")?);
                rows_from.push(None);
            }
            "-rows-from" => {
                let raw = next("-rows-from")?;
                let index: usize = raw
                    .parse()
                    .map_err(|_| anyhow!("-rows-from expects a 0-based index, got {raw}"))?;
                match rows_from.last_mut() {
                    Some(slot @ None) => *slot = Some(index),
                    Some(Some(_)) => bail!("second -rows-from for the same -m"),
                    None => {
                        bail!("-rows-from with no -m before it: -rows-from follows the -m it names")
                    }
                }
            }
            // `-rows <index> -f <container> <path>`: whose rows the output
            // that follows holds, by position among the line's `-m` flags -
            // a module's own rows, or a rows module's output. It precedes
            // its output the way `-map` does, and one document a line needs
            // none: it is the only rows there are to write.
            "-rows" => {
                let raw = next("-rows")?;
                let index: usize = raw
                    .parse()
                    .map_err(|_| anyhow!("-rows expects a 0-based index, got {raw}"))?;
                if pending_rows.is_some() {
                    bail!("two -rows options in a row: each names the output that follows it");
                }
                pending_rows = Some(index);
            }
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
            "-rows-in" => {
                if rows_in.is_some() {
                    bail!("second -rows-in specified");
                }
                rows_in = Some(resolve_input_path(&next("-rows-in")?)?);
            }
            "-jobs" => {
                let raw = next("-jobs")?;
                let parsed: usize = raw
                    .parse()
                    .map_err(|_| anyhow!("-jobs expects an integer, got {raw}"))?;
                if parsed < 1 {
                    bail!("-jobs must be >= 1, got {raw}");
                }
                jobs = Some(parsed);
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
                let rows = pending_rows.take();
                let output = OutputSpec {
                    target: pending_map.take(),
                    kind,
                    path: resolve_output_path(other)?,
                    rows,
                    spelling: format!("-f {taken} {other}"),
                };
                if rows.is_some() && !output.rows_bearing() {
                    bail!(
                        "{}: -rows names whose rows a document holds, and this output \
                         writes none",
                        output.spelling
                    );
                }
                outputs.push(output);
            }
        }
    }

    if let Some(target) = pending_map {
        bail!("-map [{target}] names no output: an output path must follow it");
    }
    if let Some(index) = pending_rows {
        bail!("-rows {index} names no output: an output path must follow it");
    }
    // A packet source takes no -i input at all, so the ordinary "no input
    // specified" refusal cannot be decided here - it needs to know whether
    // the module is one, which means opening it; `run` decides it instead.
    if outputs.is_empty() {
        bail!("no output specified");
    }
    if annotations.output && !outputs.iter().any(|o| o.kind == OutputKind::Frames) {
        bail!(
            "-annotations {ANNOTATIONS_OUT} needs an -f {EDGE_FORMAT} output: \
             the annotation stream rides beside the frames"
        );
    }

    let (modules, rows_chain) = build_modules(modules, rows_from, params, wiring, &outputs)?;

    let stream_info = match stream_info_path {
        Some(path) => Some(load_stream_info(&path)?),
        None => None,
    };

    Ok(Args {
        inputs,
        pads,
        modules,
        stream_info,
        outputs,
        annotations,
        rows_in,
        rows_chain,
        jobs,
    })
}

/// The `-m` table and its wiring, checked against the rest of the command
/// line, and the rows-module chain riding alongside it. A `-filter_complex`
/// is what makes this a network; without one the single-module spelling
/// holds and every option that only a network can use is refused by name.
fn build_modules(
    modules: Vec<String>,
    rows_from: Vec<Option<usize>>,
    params: Option<String>,
    wiring: Option<String>,
    outputs: &[OutputSpec],
) -> Result<(Modules, Vec<RowsModuleSpec>)> {
    // A network may be built entirely of nodes the host answers for, and
    // binds nothing; the wiring is then what says there is anything to run.
    if modules.is_empty() && wiring.is_none() {
        bail!("no module specified (-m)");
    }

    // The pre-existing standalone rows module: one bare -m, no
    // -filter_complex and no -rows-from at all, its rows read through
    // -rows-in rather than a stream. It is untouched by anything below.
    let standalone = modules.len() == 1 && wiring.is_none() && rows_from[0].is_none();

    // Every -m on the line splits into the stream modules - the short
    // form's own, or a network's name=path table - and the rows modules
    // chained after them, each an -m <path> naming, by position, which
    // earlier -m's rows it reads.
    let total = modules.len();
    let mut stream_raws: Vec<String> = Vec::new();
    let mut rows_chain: Vec<RowsModuleSpec> = Vec::new();
    for (position, (raw, source)) in modules.into_iter().zip(rows_from).enumerate() {
        match source {
            None => {
                // A bare path (no name=path table entry) that turns out to
                // be a rows module has nothing to read a stream from, and
                // needs a -rows-from to say whose rows it reads instead.
                if !standalone
                    && !raw.contains('=')
                    && ffrwd_wasm_runtime::runtime::exports_rows_module(&raw)
                        .with_context(|| format!("opening module {raw}"))?
                {
                    bail!(
                        "{raw}: a rows module on this line needs -rows-from <index>, naming \
                         which earlier -m's rows it reads"
                    );
                }
                stream_raws.push(raw);
            }
            Some(source) => {
                if source >= total {
                    bail!("{raw}: -rows-from {source} names no -m on this line; {total} are given");
                }
                if source >= position {
                    bail!(
                        "{raw}: -rows-from {source} names its own slot or a later one; a rows \
                         module reads an earlier -m's rows"
                    );
                }
                rows_chain.push(RowsModuleSpec {
                    path: raw,
                    slot: position,
                    source,
                });
            }
        }
    }

    for output in outputs {
        if let Some(source) = output.rows {
            if source >= total {
                bail!(
                    "{}: -rows {source} names no -m on this line; {total} are given",
                    output.spelling
                );
            }
        }
    }
    check_every_rows_module_is_read(&rows_chain, outputs)?;

    let Some(wiring) = wiring else {
        if stream_raws.len() > 1 {
            bail!(
                "{} modules were given with -m and no -filter_complex wires them",
                stream_raws.len()
            );
        }
        if let Some(output) = outputs.iter().find(|o| o.target.is_some()) {
            let target = output.target.as_deref().unwrap_or_default();
            bail!("-map [{target}] names a label of a network, and no -filter_complex wires one");
        }
        let path = stream_raws.into_iter().next().expect("one module");
        // A packet source writes one output per catalog track, so the
        // one-output-per-format rule below - built for a filter's single
        // stream - does not hold for it; every other module still gets it.
        let is_source = ffrwd_wasm_runtime::runtime::exports_packet_source(&path)
            .with_context(|| format!("opening module {path}"))?;
        if !is_source {
            check_one_output_per_format(outputs)?;
        }
        return Ok((
            Modules::Single {
                path,
                params: params.unwrap_or_default(),
            },
            rows_chain,
        ));
    };

    if params.is_some() {
        bail!(
            "-params names the parameters of one module; a network writes each node's \
             parameters into -filter_complex"
        );
    }
    if let Some(output) = outputs.iter().find(|o| o.target.is_none()) {
        bail!(
            "the -f {} output has no -map naming which of the network's labels it writes",
            output.kind.format()
        );
    }
    let bindings = stream_raws
        .iter()
        .map(|raw| resolve_binding(raw))
        .collect::<Result<Vec<_>>>()?;
    Ok((Modules::Network { bindings, wiring }, rows_chain))
}

/// Every rows module on the line has to reach an output: one whose `-rows`
/// names it, or a hop feeding one. A line writing ONE rows document and
/// naming none takes the whole chain, so nothing is left over there; a line
/// naming any is checked hop by hop.
fn check_every_rows_module_is_read(specs: &[RowsModuleSpec], outputs: &[OutputSpec]) -> Result<()> {
    let documents: Vec<&OutputSpec> = outputs.iter().filter(|o| o.rows_bearing()).collect();
    if specs.is_empty() || documents.is_empty() {
        return Ok(());
    }
    if documents.len() == 1 && documents[0].rows.is_none() {
        return Ok(());
    }
    let mut read: Vec<String> = Vec::new();
    for document in &documents {
        if let Some(source) = document.rows {
            read.extend(rows_hops(source, specs));
        }
    }
    match specs.iter().find(|spec| !read.contains(&spec.path)) {
        Some(spec) => bail!(
            "{}: this rows module's rows reach no output; -rows names which document holds them",
            spec.path
        ),
        None => Ok(()),
    }
}

/// A single module writes one stream, so it fills one output of each format.
/// A network names a label per output and may write several of the same
/// format, one per label.
///
/// Rows are the exception either way: a line writes as many rows documents as
/// it has rows to write, so two of them share a format as long as `-rows`
/// says whose rows each holds.
fn check_one_output_per_format(outputs: &[OutputSpec]) -> Result<()> {
    for (index, output) in outputs.iter().enumerate() {
        let format = output.kind.format();
        let clash = outputs[..index]
            .iter()
            .any(|o| o.kind == output.kind && (!output.rows_bearing() || o.rows == output.rows));
        if clash {
            bail!("second -f {format} output specified: at most one is allowed");
        }
    }
    Ok(())
}

enum InputReader {
    Stdin(io::Stdin),
    File(File),
}

/// Windows named-pipe codes for "the other end is gone": ERROR_BROKEN_PIPE,
/// ERROR_NO_DATA and ERROR_PIPE_NOT_CONNECTED. A pipe reports its writer's
/// exit this way rather than with the empty read a file or a stdin closes
/// with, and not all of them carry an `ErrorKind` of their own.
const PIPE_CLOSED: &[i32] = &[109, 232, 233];

/// Whether this read error means the stream ended rather than broke.
fn is_pipe_eof(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::BrokenPipe | io::ErrorKind::NotConnected
    ) || e
        .raw_os_error()
        .is_some_and(|code| PIPE_CLOSED.contains(&code))
}

impl Read for InputReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            InputReader::Stdin(r) => r.read(buf),
            InputReader::File(r) => match r.read(buf) {
                // The end of a pipe is the end of the input; a producer that
                // died before it is the plan runner's to notice, by its exit
                // code, exactly as a closed stdin already is.
                Err(e) if is_pipe_eof(&e) => Ok(0),
                other => other,
            },
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

/// Whether `path` names a Windows named pipe: one already made by whoever
/// reads it, so it is opened rather than created. A POSIX fifo is an
/// ordinary path and needs no such spelling.
fn is_named_pipe(path: &str) -> bool {
    cfg!(windows) && (path.starts_with(r"\\.\pipe\") || path.starts_with(r"\\?\pipe\"))
}

/// Outputs always create/truncate; `-y` is accepted for compatibility but
/// changes nothing. A named pipe is opened for writing instead: creating one
/// is not a thing to do to a pipe that is already there.
fn open_output(path: &OutputPath) -> Result<OutputWriter> {
    match path {
        OutputPath::Stdout => Ok(OutputWriter::Stdout(io::stdout())),
        OutputPath::File(p) if is_named_pipe(p) => std::fs::OpenOptions::new()
            .write(true)
            .open(p)
            .map(OutputWriter::File)
            .with_context(|| format!("opening output {p}")),
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
    /// The rows module chain this sink's rows flow through before they are
    /// written, when one is riding this process - a `-f ndjson`, `-f srt`
    /// or `-f webvtt` output only, and never the edge (`self.frames`)
    /// output's own annotation stream.
    rows_chain: Option<rows_chain::RowsChain>,
}

impl Sink {
    fn open(
        output: &OutputSpec,
        node: usize,
        stream: &nut::Stream,
        annotations: bool,
        rows_chain: Option<rows_chain::RowsChain>,
    ) -> Result<Sink> {
        let mut sink = Sink {
            node,
            frames: None,
            rows: None,
            subtitles: None,
            annotations: annotations && output.kind == OutputKind::Frames,
            last_pts: None,
            time_base: stream.time_base,
            rows_chain,
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
        // The rows a rows/subtitle output would write, gathered across this
        // whole call so a rows-module chain sees them as one batch rather
        // than one row at a time.
        let mut batch: Vec<String> = Vec::new();
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
            if self.rows.is_some() || self.subtitles.is_some() {
                batch.extend(stamped);
            }
        }
        if self.rows.is_some() || self.subtitles.is_some() {
            let routed = match &mut self.rows_chain {
                Some(chain) => chain.process(batch)?,
                None => batch,
            };
            if let Some(w) = self.rows.as_mut() {
                write_rows(w, &routed)?;
            }
            if let Some(w) = self.subtitles.as_mut() {
                w.push(&routed)?;
            }
        }
        Ok(())
    }

    /// The rows the module had no frame to put them on. On the annotation
    /// stream they are one record after every frame's; in an ndjson output
    /// they are ordinary lines at the end. They carry no `pts` or `time`
    /// stamp: there is no frame here to take one from. A rows-module chain
    /// reads them the same as any other batch, matching rows routed to
    /// `process` and the rest passed through.
    fn write_trailing(&mut self, rows: &[String]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        if let Some(w) = self.frames.as_mut() {
            if self.annotations {
                w.write_trailing(self.last_pts.unwrap_or(0), rows)?;
            }
        }
        let owned;
        let routed: &[String] = match &mut self.rows_chain {
            Some(chain) => {
                owned = chain.process(rows.to_vec())?;
                &owned
            }
            None => rows,
        };
        if let Some(w) = self.rows.as_mut() {
            write_rows(w, routed)?;
        }
        if let Some(w) = self.subtitles.as_mut() {
            w.push(routed)?;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        if let Some(w) = self.frames.as_mut() {
            w.finish()?;
        }
        // Whatever a rows-module chain held back across every `process`
        // call, appended once here, after every other row this sink wrote.
        if let Some(chain) = self.rows_chain.as_mut() {
            let trailing = chain.finish()?;
            if !trailing.is_empty() {
                if let Some(w) = self.rows.as_mut() {
                    write_rows(w, &trailing)?;
                }
                if let Some(w) = self.subtitles.as_mut() {
                    w.push(&trailing)?;
                }
            }
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

/// The hops one output's rows flow through, in the order they run: the rows
/// module `source` names, behind every hop feeding it, walked back through
/// each one's own `-rows-from`. Empty where `source` names a module that
/// reads its rows off frames - the rows are that module's own, untouched.
fn rows_hops(source: usize, specs: &[RowsModuleSpec]) -> Vec<String> {
    let mut hops: Vec<String> = Vec::new();
    let mut at = source;
    while let Some(spec) = specs.iter().find(|spec| spec.slot == at) {
        hops.push(spec.path.clone());
        at = spec.source;
    }
    hops.reverse();
    hops
}

/// One sink per output, each bound to the node it writes and opened against
/// that node's stream header.
///
/// A rows-bearing output (`-f ndjson`, `-f srt` or `-f webvtt`) holds the
/// rows its own `-rows` names, flowing through the hops between that module
/// and the frames they were read off. Without a `-rows` it holds the rows of
/// the node it maps -- except on a line writing ONE rows document, where it
/// takes the whole chain, which is the spelling from before `-rows` existed.
/// Each output opens its own instances of the hops its rows flow through.
fn open_sinks(args: &Args, nodes: &[usize], streams: &[nut::Stream]) -> Result<Vec<Sink>> {
    let rows_bearing = args.outputs.iter().filter(|o| o.rows_bearing()).count();
    if !args.rows_chain.is_empty() && rows_bearing == 0 {
        bail!(
            "a rows module chains onto -f ndjson, srt or webvtt, and this line writes none of \
             those"
        );
    }
    let whole: Vec<String> = args
        .rows_chain
        .iter()
        .map(|spec| spec.path.clone())
        .collect();
    let mut sinks = Vec::with_capacity(args.outputs.len());
    for ((output, node), stream) in args.outputs.iter().zip(nodes).zip(streams) {
        let hops = match (output.rows_bearing(), output.rows) {
            (true, Some(source)) => rows_hops(source, &args.rows_chain),
            (true, None) if rows_bearing == 1 => whole.clone(),
            _ => Vec::new(),
        };
        let chain = if hops.is_empty() {
            None
        } else {
            Some(rows_chain::RowsChain::open(&hops)?)
        };
        sinks.push(Sink::open(
            output,
            *node,
            stream,
            args.annotations.output,
            chain,
        )?);
    }
    Ok(sinks)
}

/// Reads every input a frame at a time into the scheduler, then waits for
/// every lane to drain. An input that has ended hands over its trailing rows
/// and is left alone; the workers do everything else.
fn run_lanes(
    net: Network,
    mut readers: Vec<Input>,
    formats: &[Format],
    sinks: Vec<Sink>,
    jobs: Option<usize>,
) -> Result<()> {
    let workers = scheduler::worker_count(jobs);
    let sched = scheduler::Scheduler::start(net.into_seeds(), sinks, readers.len(), workers);

    let mut feed = || -> Result<bool> {
        let mut buf: Vec<u8> = Vec::new();
        let mut open = vec![true; readers.len()];
        let mut index = vec![0u64; readers.len()];
        loop {
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
                        let frame = Frame {
                            pts,
                            data: Arc::new(std::mem::take(&mut buf)),
                            rows: reader.take_rows(pts),
                        };
                        index[input] += 1;
                        any = true;
                        if !sched.push_input(input, frame) {
                            return Ok(false);
                        }
                    }
                    None => {
                        open[input] = false;
                        // The trailing record sits past the last frame, so it
                        // is in hand exactly now.
                        let trailing = reader.take_trailing();
                        if !sched.input_eof(input, &trailing) {
                            return Ok(false);
                        }
                    }
                }
            }
            if !any {
                return Ok(true);
            }
        }
    };
    match feed() {
        Ok(_) => sched.finish(),
        Err(e) => sched.abort(e),
    }
}

/// Opens the module once for the stream it will read.
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
    // A packet source rides alone, ahead of every other check: it produces
    // its own packets and takes no -i input, so the "no input specified"
    // refusal just below does not apply to it.
    if let Modules::Single { path, params } = &args.modules {
        let is_packet_source = ffrwd_wasm_runtime::runtime::exports_packet_source(path)
            .with_context(|| format!("opening module {path}"))?;
        if is_packet_source {
            if !args.inputs.is_empty() {
                bail!(
                    "{path} is a packet source: it produces its own packets and reads no -i \
                     input; this command gives it {}",
                    args.inputs.len()
                );
            }
            return run_packet_source(args, path, params);
        }
    }

    // A rows module rides alone too: no stream feeds it, so it takes no -i
    // either, and its rows arrive through -rows-in instead.
    if let Modules::Single { path, params } = &args.modules {
        let is_rows_module = ffrwd_wasm_runtime::runtime::exports_rows_module(path)
            .with_context(|| format!("opening module {path}"))?;
        if is_rows_module {
            if !args.inputs.is_empty() {
                bail!(
                    "{path} is a rows module: it reads no stream, and this command gives it {} \
                     -i input(s); rows arrive through -rows-in instead",
                    args.inputs.len()
                );
            }
            let rows_in = args.rows_in.as_ref().ok_or_else(|| {
                anyhow!("{path} is a rows module: it reads rows through -rows-in, which this command does not give it")
            })?;
            return run_rows_module(args, path, params, rows_in);
        }
    }
    // -rows-in names a rows module's input; every other shape reads neither
    // a rows module's -i (none) nor -rows-in (nothing) - so a -rows-in this
    // far along names something that is not one.
    if args.rows_in.is_some() {
        match &args.modules {
            Modules::Single { path, .. } => {
                bail!("-rows-in follows a rows module's -m; {path} is not one")
            }
            Modules::Network { .. } => bail!(
                "-rows-in names a rows module's input; a network wires modules together and \
                 reads none itself"
            ),
        }
    }

    if args.inputs.is_empty() {
        bail!("no input specified (-i)");
    }

    // A packet sink is dispatched before any header is read here: its
    // reader threads open the inputs themselves and drain them from the
    // first byte, so no producer ever waits on another producer's warmup
    // or on the module's own connect.
    if let Modules::Single { path, params } = &args.modules {
        let is_packet_sink = ffrwd_wasm_runtime::runtime::exports_packet_sink(path)
            .with_context(|| format!("opening module {path}"))?;
        if is_packet_sink {
            return run_packet_sink(args, path, params);
        }
    }

    // -pad names a packet sink's row and rendition for one of its -i; every
    // other module has neither to carry.
    if args.pads.iter().any(Option::is_some) {
        bail!("-pad follows a packet sink's -i; this module is not a packet sink");
    }

    // Headers are read concurrently: one producer's first bytes can wait on
    // another producer's whole chain starting up, and a fast input left
    // unread meanwhile fills its pipe and blocks the producer they share.
    let mut readers: Vec<Input> = Vec::new();
    std::thread::scope(|scope| -> Result<()> {
        let handles: Vec<_> = args
            .inputs
            .iter()
            .map(|path| {
                scope.spawn(move || -> Result<Input> {
                    let reader = io::BufReader::with_capacity(1 << 20, open_input(path)?);
                    if args.annotations.input {
                        nut::Demuxer::open_annotated(reader)
                    } else {
                        nut::Demuxer::open(reader)
                    }
                    .context("reading the NUT input")
                })
            })
            .collect();
        for handle in handles {
            let demuxer = handle
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic))?;
            readers.push(demuxer);
        }
        Ok(())
    })?;

    if let Modules::Single { .. } = &args.modules {
        // A sink is what reads several streams off several -i; a frame module
        // reads its pads out of ONE input, wired by -filter_complex.
        if readers.len() > 1 {
            bail!("a single frame module reads one stream; a second -i needs a -filter_complex");
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
            let filter = open_filter(path, params, &formats[0], &infos[0])?;
            check_reads_rows(args, &filter)?;
            let reopen = network::reopen_for(&filter, path, params, &formats[0], &infos[0]);
            let net = Network::single(filter, &formats[0], reopen);
            let nodes = vec![0usize; args.outputs.len()];
            let sink_streams = vec![streams[0].clone(); args.outputs.len()];
            let sinks = open_sinks(args, &nodes, &sink_streams)?;
            run_lanes(net, readers, &formats, sinks, args.jobs)
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
            let sinks = open_sinks(args, &nodes, &sink_streams)?;
            run_lanes(net, readers, &formats, sinks, args.jobs)
        }
    }
}

/// The encoded inputs of a packet sink through ONE instance: packets handed
/// through untouched, in decode order per pad, rows to the row outputs. No
/// frames leave, so the only outputs are rows and null.
///
/// Pads run independently. Packets are not frames: nothing pairs a pad's
/// packet with another pad's, so there is no lockstep to hold and a call may
/// carry packets on some pads and none on others. One reader thread per pad
/// takes blocking reads off its pipe into a byte-bounded queue, and the
/// drive loop hands the module whatever has arrived: ONE producer feeding
/// pads at unequal packet rates interleaves its writes by dts, so a loop
/// that waited on a specific pad would deadlock against the producer's own
/// blocking write once the other pads' pipes filled. The queue bound is the
/// flow control: a stalled consumer stops the producer instead of buffering
/// it without limit. The wasm instance is called from this thread alone;
/// only the I/O grows threads.
fn run_packet_sink(args: &Args, module: &str, params: &str) -> Result<()> {
    // A packet sink is an exclusive lane by nature: packets reach it in
    // decode order, so one instance reads them and `-jobs` caps nothing.
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

    // The readers start before anything else: each opens its own input and
    // pumps packets into its bounded queue from the first byte, so a fast
    // producer (a live microphone, say) is drained while a slow one's
    // whole chain still warms up, and while the module's own open - which
    // may dial a relay - takes its time. The threads are not joined: on an
    // error the process exits and takes a reader blocked in a pipe read
    // with it, which a join would wait on forever.
    let pads = args.inputs.len();
    let queues = Arc::new(PadQueues::new(pads));
    let (told, headers) = std::sync::mpsc::channel();
    for (pad, path) in args.inputs.iter().enumerate() {
        let queues = Arc::clone(&queues);
        let told = told.clone();
        let path = path.clone();
        std::thread::spawn(move || {
            let opened = (|| {
                let reader = io::BufReader::with_capacity(1 << 20, open_input(&path)?);
                // -annotations input is refused above, so the packets are bare.
                nut::Demuxer::open(reader).context("reading the NUT input")
            })();
            match opened {
                Ok(input) => {
                    let _ = told.send((pad, Ok(input.stream().clone())));
                    read_pad(input, pad, &queues);
                }
                Err(err) => {
                    let _ = told.send((pad, Err(err)));
                }
            }
        });
    }
    drop(told);

    // The headers arrive in whatever order the producers start; the module
    // is opened once every pad has reported its stream.
    let mut streams: Vec<Option<nut::Stream>> = (0..pads).map(|_| None).collect();
    for _ in 0..pads {
        let (pad, stream) = headers.recv().expect("every reader reports its header");
        streams[pad] = Some(stream.with_context(|| format!("input {pad}"))?);
    }
    let mut sink_inputs = Vec::with_capacity(pads);
    for (pad, stream) in streams.iter().enumerate() {
        let stream = stream.as_ref().expect("every pad reported");
        let (row, rendition) = resolve_pad(&args.pads, pad);
        sink_inputs.push(coded_pad(args, module, pad, stream, row, rendition.into())?);
    }
    let mut sink = runtime::PacketSink::open(module, &sink_inputs, params)
        .with_context(|| format!("opening module {module}"))?;

    let outcome = (|| -> Result<()> {
        loop {
            let (carried, done) = queues.take()?;
            if done {
                return Ok(());
            }
            let emitted = sink
                .process(&carried, false)
                .with_context(|| format!("{}: processing packets", sink.name()))?;
            for writer in &mut row_outputs {
                write_rows(writer, &emitted.rows)?;
            }
        }
    })();
    // A reader still waiting for queue space must wake and stop.
    queues.close();
    outcome?;

    // The final call, carrying no packets: whatever the sink held back
    // arrives as rows, and the trailing rows follow them.
    let emitted = sink
        .process(&vec![Vec::new(); pads], true)
        .with_context(|| format!("{}: the final call", sink.name()))?;
    for writer in &mut row_outputs {
        write_rows(writer, &emitted.rows)?;
        write_rows(writer, &emitted.trailing)?;
        writer.flush()?;
    }
    Ok(())
}

/// `-rows-in`: one JSON object per line, blank lines skipped. No stream
/// carries these, so nothing paces them the way a producer paces a NUT
/// input - the whole file is read before the module sees any of it.
fn read_ndjson_rows(reader: InputReader) -> Result<Vec<String>> {
    io::BufReader::new(reader)
        .lines()
        .filter(|line| !matches!(line, Ok(l) if l.trim().is_empty()))
        .collect::<io::Result<Vec<String>>>()
        .context("reading -rows-in")
}

/// A rows module through ONE instance: no stream at all, so nothing paces
/// it the way a frame or a packet does. `-rows-in`'s whole file is read
/// first, then `process` is called once with every row, then `finish` -
/// there is no batching to choose, since a rows module is handed no
/// producer to batch against.
fn run_rows_module(
    args: &Args,
    module_path: &str,
    params: &str,
    rows_in: &InputPath,
) -> Result<()> {
    if args.annotations.input || args.annotations.output {
        bail!(
            "{module_path}: a rows module reads no stream and writes none, so -annotations has \
             nothing to carry rows on"
        );
    }

    let mut row_outputs: Vec<RowOutput> = Vec::new();
    for output in &args.outputs {
        match output.kind {
            OutputKind::Rows => row_outputs.push(open_row_output(&output.path)?),
            OutputKind::Null => {}
            _ => bail!(
                "{}: a rows module emits rows alone; its outputs are -f {ROWS_FORMAT} and -f null",
                output.spelling
            ),
        }
    }

    let rows = read_ndjson_rows(open_input(rows_in)?)?;

    let mut module = runtime::RowsModule::open(module_path, params)
        .with_context(|| format!("opening module {module_path}"))?;
    let mut emitted = module
        .process(&rows)
        .with_context(|| format!("{}: processing rows", module.name()))?;
    emitted.extend(
        module
            .finish()
            .with_context(|| format!("{}: finish", module.name()))?,
    );

    for writer in &mut row_outputs {
        write_rows(writer, &emitted)?;
        writer.flush()?;
    }
    Ok(())
}

/// How many buffered bytes one pad's reader may hold before it waits for
/// the drive loop to drain. The one producer interleaves its pads by dts,
/// so while it writes a slow pad's next packet the fast pads' packets keep
/// arriving and must land somewhere; 8 MB holds a couple of seconds of a
/// high-bitrate rung, which covers that skew, and is small enough that a
/// stalled consumer stops the producer instead of buffering it without
/// limit. A packet larger than the whole bound still crosses once its queue
/// is empty, so no single packet can wedge a pad.
const PAD_BUFFER_BYTES: usize = 8 << 20;

/// One pad's queue between its reader thread and the drive loop.
#[derive(Default)]
struct PadQueue {
    packets: Vec<runtime::Packet>,
    bytes: usize,
    /// The pad's input ended; set after its last packet is queued.
    closed: bool,
    /// The pad's read failed; the drive loop raises it.
    failed: Option<anyhow::Error>,
}

/// What the reader threads and the drive loop share: the per-pad queues
/// under one lock, and a condvar for each direction of waiting.
struct PadQueues {
    state: Mutex<PadState>,
    /// A pad gained a packet, closed, or failed.
    filled: Condvar,
    /// The drive loop drained, freeing space - or is gone for good.
    drained: Condvar,
}

struct PadState {
    pads: Vec<PadQueue>,
    /// The drive loop has stopped consuming; readers stop instead of
    /// waiting for space that will never come.
    dead: bool,
}

impl PadQueues {
    fn new(pads: usize) -> Self {
        PadQueues {
            state: Mutex::new(PadState {
                pads: (0..pads).map(|_| PadQueue::default()).collect(),
                dead: false,
            }),
            filled: Condvar::new(),
            drained: Condvar::new(),
        }
    }

    /// Everything queued so far, per pad, waiting until at least one pad
    /// has packets. `(_, true)` once every pad is closed and drained; a
    /// pad's stored read error is raised here, on the drive loop's thread.
    fn take(&self) -> Result<(Vec<Vec<runtime::Packet>>, bool)> {
        let mut state = self.state.lock().expect("a reader panicked with the lock");
        loop {
            if let Some(queue) = state.pads.iter_mut().find(|q| q.failed.is_some()) {
                return Err(queue.failed.take().expect("found by is_some"));
            }
            if state.pads.iter().any(|q| !q.packets.is_empty()) {
                break;
            }
            if state.pads.iter().all(|q| q.closed) {
                return Ok((Vec::new(), true));
            }
            state = self
                .filled
                .wait(state)
                .expect("a reader panicked with the lock");
        }
        let carried = state
            .pads
            .iter_mut()
            .map(|queue| {
                queue.bytes = 0;
                std::mem::take(&mut queue.packets)
            })
            .collect();
        self.drained.notify_all();
        Ok((carried, false))
    }

    /// The drive loop is done, normally or not: wake every waiting reader
    /// so it can stop.
    fn close(&self) {
        self.state
            .lock()
            .expect("a reader panicked with the lock")
            .dead = true;
        self.drained.notify_all();
    }
}

/// Settles packet durations for one pad. NUT frames carry no duration
/// field, so the wire never says it directly; where the stream does not
/// reorder (decode delay 0) decode order is presentation order, and the
/// next packet's pts settles the previous one's duration exactly. A
/// reordering stream's presentation successor is not the next packet read,
/// and the final packet has no successor at all: both stay None.
struct Durations {
    /// Whether successive pts settle durations at all.
    settled: bool,
    /// The packet the next one's pts will settle.
    pending: Option<runtime::Packet>,
}

impl Durations {
    fn new(decode_delay: u64) -> Durations {
        Durations {
            settled: decode_delay == 0,
            pending: None,
        }
    }

    /// Takes one packet in and returns the packet now ready to queue: the
    /// previous one with its duration settled, or - for a stream whose
    /// durations stay unknown - this one straight through.
    fn push(&mut self, packet: runtime::Packet) -> Option<runtime::Packet> {
        if !self.settled {
            return Some(packet);
        }
        let mut ready = self.pending.replace(packet);
        if let Some(prev) = ready.as_mut() {
            let next_pts = self.pending.as_ref().expect("just replaced").pts;
            // Unknown is spelled None, never 0, so a step that is not
            // forward settles nothing.
            prev.duration = Some(next_pts - prev.pts).filter(|d| *d > 0);
        }
        ready
    }

    /// The held packet at the end of the input, its duration unknown: no
    /// successor settles it.
    fn finish(&mut self) -> Option<runtime::Packet> {
        self.pending.take()
    }
}

/// One pad's reader: blocking reads off its own input, each packet into the
/// pad's queue, waiting whenever the queue is over its byte bound. Decode
/// order per pad is preserved by construction - one thread, one queue.
fn read_pad(mut input: Input, pad: usize, queues: &PadQueues) {
    let mut durations = Durations::new(input.stream().decode_delay);
    let mut buf: Vec<u8> = Vec::new();
    let mut index = 0u64;
    loop {
        let read = input
            .read_packet(&mut buf)
            .with_context(|| format!("reading packet {index} of pad {pad}"));
        match read {
            Ok(Some(packet)) => {
                let ready = durations.push(runtime::Packet {
                    pts: packet.pts,
                    dts: packet.dts,
                    duration: None,
                    keyframe: packet.keyframe,
                    data: std::mem::take(&mut buf),
                });
                index += 1;
                if let Some(packet) = ready {
                    if !queue_packet(queues, pad, packet) {
                        return;
                    }
                }
            }
            Ok(None) => {
                if let Some(packet) = durations.finish() {
                    if !queue_packet(queues, pad, packet) {
                        return;
                    }
                }
                let mut state = queues.state.lock().expect("the drive loop holds no panic");
                state.pads[pad].closed = true;
                queues.filled.notify_one();
                return;
            }
            Err(error) => {
                let mut state = queues.state.lock().expect("the drive loop holds no panic");
                state.pads[pad].failed = Some(error);
                state.pads[pad].closed = true;
                queues.filled.notify_one();
                return;
            }
        }
    }
}

/// One packet into its pad's queue, waiting whenever the queue is over its
/// byte bound. False when the drive loop is gone and the reader must stop.
fn queue_packet(queues: &PadQueues, pad: usize, packet: runtime::Packet) -> bool {
    let mut state = queues.state.lock().expect("the drive loop holds no panic");
    while !state.dead
        && state.pads[pad].bytes >= PAD_BUFFER_BYTES
        && !state.pads[pad].packets.is_empty()
    {
        state = queues
            .drained
            .wait(state)
            .expect("the drive loop holds no panic");
    }
    if state.dead {
        return false;
    }
    state.pads[pad].bytes += packet.data.len();
    state.pads[pad].packets.push(packet);
    // One consumer waits on `filled`, so one wake reaches it.
    queues.filled.notify_one();
    true
}

/// One pad of a packet sink, from the NUT header the input opened with. A
/// stream this wire carries decoded never reaches a sink: the encoder lives
/// in the ffmpeg on the other side of the pipe.
fn coded_pad(
    args: &Args,
    module: &str,
    pad: usize,
    stream: &nut::Stream,
    row: u32,
    rendition: runtime::RenditionMeta,
) -> Result<runtime::SinkInput> {
    let Some(codec) = stream.codec_name() else {
        let carried = if let Some(pix_fmt) = stream.pix_fmt() {
            format!("decoded {pix_fmt} video")
        } else if let Some(sample_fmt) = stream.sample_fmt() {
            format!("{sample_fmt} audio")
        } else {
            format!("codec tag {}", stream.fourcc_name())
        };
        bail!(
            "{module} consumes encoded packets, and input {pad} carries {carried}; \
             put it after the encoder (ffmpeg ... -c:v <codec> -c:a <codec> -f nut)"
        );
    };
    let format = match stream.media {
        nut::Media::Video {
            width,
            height,
            sample_width,
            sample_height,
            ..
        } => runtime::CodedFormat::Video {
            width,
            height,
            sample_aspect_ratio: aspect_from(sample_width, sample_height),
            color: color_from(&stream.media),
        },
        nut::Media::Audio {
            sample_rate,
            channels,
        } => runtime::CodedFormat::Audio {
            sample_rate,
            channels,
            // The NUT audio header carries no layout; see `format_from_stream`.
            channel_layout: None,
        },
    };
    // Profile and level: the NUT stream header has no field for either, so
    // h264's are read off the SPS the extradata carries. The other codecs
    // this wire names keep theirs elsewhere (hevc and av1 inside their own
    // headers, aac nowhere), and stay None rather than guessed.
    let (profile, level) = match codec {
        "h264" => match h264_profile_level(&stream.extradata) {
            Some((profile, level)) => (Some(profile), Some(level)),
            None => (None, None),
        },
        _ => (None, None),
    };
    let coded = runtime::CodedStream {
        codec: codec.to_string(),
        time_base: TimeBase {
            num: stream.time_base.num,
            den: stream.time_base.den,
        },
        format,
        extradata: stream.extradata.clone(),
        profile,
        level,
    };
    let info = args.stream_info.clone().unwrap_or_else(|| StreamInfo {
        index: pad as u32,
        kind: stream.kind().to_string(),
        codec: codec.to_string(),
        duration: None,
        tags: Vec::new(),
    });
    Ok(runtime::SinkInput {
        stream: coded,
        info,
        row,
        rendition,
    })
}

/// This pad's row and rendition, as `-pad` said them, or the defaults a pad
/// with none gets: its own index among the sink's inputs, and a rendition
/// with every field None.
fn resolve_pad(pads: &[Option<PadSpec>], pad: usize) -> (u32, PadRendition) {
    match pads.get(pad).and_then(Option::as_ref) {
        Some(spec) => (spec.row.unwrap_or(pad as u32), spec.rendition.clone()),
        None => (pad as u32, PadRendition::default()),
    }
}

impl From<PadRendition> for runtime::RenditionMeta {
    fn from(r: PadRendition) -> Self {
        runtime::RenditionMeta {
            name: r.name,
            bandwidth: r.bandwidth,
            codecs: r.codecs,
            language: r.language,
        }
    }
}

/// The nut::Stream header for one packet-source track: the coded fourcc this
/// wire has a tag for, the codec's own time base and extradata, and
/// `decode_delay` as `run_packet_source`'s pull loop settled it.
fn coded_stream_for(coded: &runtime::CodedStream, decode_delay: u64) -> Result<nut::Stream> {
    let kind = coded.format.kind();
    let fourcc = nut::fourcc_for_coded(kind, &coded.codec).ok_or_else(|| {
        let names: Vec<&str> = match kind {
            "video" => nut::CODED_VIDEO_FOURCCS
                .iter()
                .map(|(name, _)| *name)
                .collect(),
            _ => nut::CODED_AUDIO_FOURCCS
                .iter()
                .map(|(name, _)| *name)
                .collect(),
        };
        anyhow!(
            "{} is not a {kind} codec this wire carries; only {} are",
            coded.codec,
            names.join(", ")
        )
    })?;
    let time_base = nut::TimeBase {
        num: coded.time_base.num,
        den: coded.time_base.den,
    };
    let max_pts_distance = time_base.den.div_ceil(time_base.num.max(1));
    let media = match &coded.format {
        runtime::CodedFormat::Video {
            width,
            height,
            sample_aspect_ratio,
            color,
        } => {
            let (sample_width, sample_height) = sample_aspect_ratio
                .and_then(|(num, den)| (num > 0 && den > 0).then_some((num as u64, den as u64)))
                .unwrap_or((1, 1));
            nut::Media::Video {
                width: *width,
                height: *height,
                sample_width,
                sample_height,
                colorspace_type: colorspace_type_for(color.as_ref()),
            }
        }
        runtime::CodedFormat::Audio {
            sample_rate,
            channels,
            ..
        } => nut::Media::Audio {
            sample_rate: *sample_rate,
            channels: *channels,
        },
    };
    Ok(nut::Stream {
        fourcc: fourcc.to_vec(),
        time_base,
        msb_pts_shift: 14,
        max_pts_distance,
        decode_delay,
        extradata: coded.extradata.clone(),
        media,
    })
}

/// The wire's colorspace code for `color`, or 0 (unknown) where there is
/// none, or the wire has no matrix for it - the reverse of `color_from`.
fn colorspace_type_for(color: Option<&runtime::ColorInfo>) -> u64 {
    let Some(color) = color else { return 0 };
    let space = match color.space {
        "bt470bg" => 1,
        "bt709" => 2,
        _ => return 0,
    };
    space | if color.range == "pc" { 16 } else { 0 }
}

/// One packet through an already-open track output. `nut::Packet.dts` is not
/// read by `write_coded`; the field only matters for `run_packet_source`'s
/// own decode_delay bookkeeping before the output is open.
fn write_coded_packet(muxer: &mut FrameOutput, packet: &runtime::Packet) -> Result<()> {
    let framed = nut::Packet {
        pts: packet.pts,
        dts: packet.dts,
        keyframe: packet.keyframe,
    };
    muxer.write_coded(&framed, &packet.data)
}

/// Opens one track's output now that its `decode_delay` has settled - see
/// `run_packet_source` - and drains whatever was buffered waiting for it.
fn settle_track(
    decode_delay: u64,
    track: &runtime::SourceTrack,
    output: &OutputSpec,
    pending: &mut Vec<runtime::Packet>,
    slot: &mut Option<FrameOutput>,
) -> Result<()> {
    let stream = coded_stream_for(&track.stream, decode_delay)?;
    let mut muxer = open_frame_output(&output.path, &stream, false)
        .with_context(|| format!("opening output {}", output.spelling))?;
    for packet in pending.drain(..) {
        write_coded_packet(&mut muxer, &packet)?;
    }
    *slot = Some(muxer);
    Ok(())
}

/// A packet source rides alone: no `-i`, one `-f nut` output per catalog
/// track, in catalog order. Packets arrive in decode order already -
/// `PacketSource::next`'s own contract - so nothing here reorders them; the
/// only work is settling each track's `decode_delay` before its output's
/// header is written, since the wit `coded-stream` a source publishes
/// carries no such field.
///
/// `packet.dts` is `None` until the wire settles it, exactly the convention
/// `nut::Packet.dts` uses for a stream this host demuxes (nut/mod.rs): so
/// the count of a track's leading `None` packets IS its decode_delay. This
/// buffers a track's packets until the first settled `dts` arrives, opens
/// that track's output with the count it took, and flushes what was held.
/// A track that never settles - every packet arrives `None` - opens once
/// the source is done, declaring a reorder as deep as every packet it saw.
///
/// Rows are a sink's alone; a source emits none in this wave.
fn run_packet_source(args: &Args, module: &str, params: &str) -> Result<()> {
    if !args.inputs.is_empty() {
        bail!(
            "{module} is a packet source: it produces its own packets and reads no -i input; \
             this command gives it {}",
            args.inputs.len()
        );
    }
    if args.annotations.input || args.annotations.output {
        bail!(
            "a packet source's outputs carry encoded packets, not frames, so -annotations has \
             nothing to give or take here"
        );
    }
    for output in &args.outputs {
        if output.kind != OutputKind::Frames {
            bail!(
                "{}: a packet source writes every track as -f {EDGE_FORMAT}; -f {} is not that",
                output.spelling,
                output.kind.format()
            );
        }
    }

    let (mut source, catalog) = runtime::PacketSource::open(module, params)
        .with_context(|| format!("opening module {module}"))?;

    if args.outputs.len() != catalog.tracks.len() {
        bail!(
            "{module}'s catalog names {} track(s) and this command was given {} output(s)",
            catalog.tracks.len(),
            args.outputs.len()
        );
    }

    let tracks = catalog.tracks.len();
    let mut pending: Vec<Vec<runtime::Packet>> = (0..tracks).map(|_| Vec::new()).collect();
    let mut muxers: Vec<Option<FrameOutput>> = (0..tracks).map(|_| None).collect();

    while let Some(pads) = source
        .next()
        .with_context(|| format!("{}: pulling packets", source.name()))?
    {
        for (index, pad) in pads.into_iter().enumerate() {
            for packet in pad.packets {
                if let Some(muxer) = muxers[index].as_mut() {
                    write_coded_packet(muxer, &packet)?;
                    continue;
                }
                let settled = packet.dts.is_some();
                pending[index].push(packet);
                if settled {
                    let decode_delay = (pending[index].len() - 1) as u64;
                    settle_track(
                        decode_delay,
                        &catalog.tracks[index],
                        &args.outputs[index],
                        &mut pending[index],
                        &mut muxers[index],
                    )?;
                }
            }
        }
    }

    for index in 0..tracks {
        if muxers[index].is_none() {
            let decode_delay = pending[index].len() as u64;
            settle_track(
                decode_delay,
                &catalog.tracks[index],
                &args.outputs[index],
                &mut pending[index],
                &mut muxers[index],
            )?;
        }
    }

    for muxer in muxers.iter_mut().flatten() {
        muxer.finish()?;
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
    /// The ffmpeg codec names a packet sink accepts for VIDEO, most preferred
    /// first, and empty for every codec. Present only for a module exporting
    /// the packet sink, which is how that export is reported - the way
    /// `window` marks a windowed module.
    #[serde(skip_serializing_if = "Option::is_none")]
    video_codecs: Option<Vec<String>>,
    /// The same for AUDIO. Empty for a sink built against a world before
    /// 0.12.0, which read no audio stream.
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_codecs: Option<Vec<String>>,
    /// How many streams of each kind a packet sink reads: "none", "one" or
    /// "many". Present only for a module exporting the packet sink.
    #[serde(skip_serializing_if = "Option::is_none")]
    video_streams: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_streams: Option<&'static str>,
    /// Whether the module exports a packet source: the same boolean-flag
    /// convention `nn`/`http`/`udp` use, false for every module built before
    /// 0.13.0.
    source: bool,
    /// Whether the module exports a rows module: no stream at all, rows in
    /// and out of one call. False for every module built before 0.14.0.
    rows_module: bool,
    /// The schema of the row a rows module reads on `process`. `null` for a
    /// module that is not one; present and possibly empty for one that is -
    /// `rows_schema` above is what it emits, this is what it reads.
    #[serde(skip_serializing_if = "Option::is_none")]
    input_rows_schema: Option<serde_json::Value>,
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

/// How a sink's per-kind stream count reads in a description.
fn streams_read(arity: ffrwd_wasm_runtime::runtime::Arity) -> &'static str {
    use ffrwd_wasm_runtime::runtime::Arity;
    match arity {
        Arity::Zero => "none",
        Arity::One => "one",
        Arity::Many => "many",
        Arity::Any => "any",
    }
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
    let has_source = ffrwd_wasm_runtime::runtime::exports_packet_source(module_path)
        .with_context(|| format!("describing {module_path}"))?;
    let has_rows_module = ffrwd_wasm_runtime::runtime::exports_rows_module(module_path)
        .with_context(|| format!("describing {module_path}"))?;

    let has_frames = has_filter || has_window;
    if !has_frames && !has_values && !has_packet && !has_source && !has_rows_module {
        let exports = ffrwd_wasm_runtime::runtime::exports(module_path)?;
        if exports.is_empty() {
            bail!("{module_path} exports nothing, so no describe is possible");
        }
        bail!(
            "{module_path} exports neither a filter, a packet sink, a packet source, a rows \
             module, nor value functions; it exports {}",
            exports.join(", ")
        );
    }
    if has_packet && has_frames {
        bail!(
            "{module_path} exports both a packet sink and a frame interface; \
             a module is one or the other"
        );
    }
    if has_source && (has_frames || has_packet) {
        bail!(
            "{module_path} exports a packet source alongside {}; a module is one or the other",
            if has_frames {
                "a frame interface"
            } else {
                "a packet sink"
            }
        );
    }
    // A rows module carries no stream at all - `values` is the one interface
    // every other kind may also export, so it is the one thing a rows module
    // may share, the way `fauxlate` does.
    if has_rows_module && (has_frames || has_packet || has_source) {
        bail!(
            "{module_path} exports a rows module alongside {}; a module reading a stream and \
             a module reading none are one or the other",
            if has_frames {
                "a frame interface"
            } else if has_packet {
                "a packet sink"
            } else {
                "a packet source"
            }
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
        video_codecs: None,
        audio_codecs: None,
        video_streams: None,
        audio_streams: None,
        source: false,
        rows_module: false,
        input_rows_schema: None,
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
        description.video_codecs = Some(described.video_codecs);
        description.audio_codecs = Some(described.audio_codecs);
        description.video_streams = Some(streams_read(described.video));
        description.audio_streams = Some(streams_read(described.audio));
    }

    if has_source {
        let described = ffrwd_wasm_runtime::runtime::describe_packet_source(module_path)
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
        description.source = true;
    }

    if has_rows_module {
        let described = ffrwd_wasm_runtime::runtime::describe_rows_module(module_path)
            .with_context(|| format!("describing {module_path}"))?;
        let meta = described.meta;
        description.params_schema = Some(parse_schema(
            &meta.params_schema,
            &meta.name,
            "params_schema",
        )?);
        description.rows_schema = Some(parse_schema(&meta.rows_schema, &meta.name, "rows_schema")?);
        description.input_rows_schema = Some(parse_schema(
            &described.input_rows_schema,
            &meta.name,
            "input_rows_schema",
        )?);
        description.rows_language = meta.rows_language;
        description.version = Some(meta.version);
        description.name = Some(meta.name);
        description.rows_module = true;
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

/// One byte of `bytes` as two lowercase hex digits, concatenated.
fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A catalog track's geometry, tagged by kind - `probe`'s JSON names the
/// variant `video` or `audio` and nests only the fields that kind declares.
#[derive(Serialize)]
#[serde(rename_all = "lowercase")]
enum CatalogFormatJson {
    Video { width: u32, height: u32 },
    Audio { sample_rate: u32, channels: u32 },
}

#[derive(Serialize)]
struct CatalogRenditionJson {
    name: Option<String>,
    bandwidth: Option<u64>,
    codecs: Option<String>,
    language: Option<String>,
}

#[derive(Serialize)]
struct CatalogTrackJson {
    codec: String,
    time_base: [u64; 2],
    format: CatalogFormatJson,
    extradata: String,
    profile: Option<i32>,
    level: Option<i32>,
    row: u32,
    rendition: CatalogRenditionJson,
}

#[derive(Serialize)]
struct CatalogJson {
    tracks: Vec<CatalogTrackJson>,
    bounded: bool,
}

/// `catalog` as `--probe` prints it: one line, tracks in catalog order.
fn catalog_json(catalog: &runtime::Catalog) -> CatalogJson {
    CatalogJson {
        bounded: catalog.bounded,
        tracks: catalog
            .tracks
            .iter()
            .map(|t| CatalogTrackJson {
                codec: t.stream.codec.clone(),
                time_base: [t.stream.time_base.num, t.stream.time_base.den],
                format: match &t.stream.format {
                    runtime::CodedFormat::Video { width, height, .. } => CatalogFormatJson::Video {
                        width: *width,
                        height: *height,
                    },
                    runtime::CodedFormat::Audio {
                        sample_rate,
                        channels,
                        ..
                    } => CatalogFormatJson::Audio {
                        sample_rate: *sample_rate,
                        channels: *channels,
                    },
                },
                extradata: to_hex(&t.stream.extradata),
                profile: t.stream.profile,
                level: t.stream.level,
                row: t.row,
                rendition: CatalogRenditionJson {
                    name: t.rendition.name.clone(),
                    bandwidth: t.rendition.bandwidth,
                    codecs: t.rendition.codecs.clone(),
                    language: t.rendition.language.clone(),
                },
            })
            .collect(),
    }
}

/// The `-params` value out of `--probe`'s trailing argv; every other flag is
/// refused by name.
fn parse_probe_args(rest: &[String]) -> Result<String> {
    let mut it = rest.iter();
    let mut params: Option<String> = None;
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-params" => {
                if params.is_some() {
                    bail!("second -params specified");
                }
                params = Some(
                    it.next()
                        .cloned()
                        .ok_or_else(|| anyhow!("-params requires a value"))?,
                );
            }
            other => bail!("--probe: unknown flag {other}"),
        }
    }
    Ok(params.unwrap_or_default())
}

/// Compiles and instantiates `module_path` as a packet source and calls its
/// `probe` - the compile-time twin of `run_packet_source`'s `open`, reading
/// the catalog without opening the source for a run. One JSON line out.
fn probe_module(module_path: &str, params: &str) -> Result<String> {
    let has_source = ffrwd_wasm_runtime::runtime::exports_packet_source(module_path)
        .with_context(|| format!("probing {module_path}"))?;
    if !has_source {
        let exports = ffrwd_wasm_runtime::runtime::exports(module_path)?;
        bail!(
            "{module_path} does not export a packet source, so it cannot be probed; it exports {}",
            if exports.is_empty() {
                "nothing".to_string()
            } else {
                exports.join(", ")
            }
        );
    }
    let catalog = runtime::PacketSource::probe(module_path, params)
        .with_context(|| format!("probing {module_path}"))?;
    serde_json::to_string(&catalog_json(&catalog)).context("serializing the catalog")
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

    if raw_args.first().map(String::as_str) == Some("--probe") {
        match raw_args.get(1) {
            None => {
                eprintln!("ffrwd-wasm: --probe requires a module path");
                std::process::exit(2);
            }
            Some(module_path) => {
                let params = match parse_probe_args(&raw_args[2..]) {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("ffrwd-wasm: {e:#}");
                        std::process::exit(2);
                    }
                };
                match probe_module(module_path, &params) {
                    Ok(json) => {
                        println!("{json}");
                        std::process::exit(0);
                    }
                    Err(e) => {
                        eprintln!("ffrwd-wasm: {e:#}");
                        std::process::exit(1);
                    }
                }
            }
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
mod stream_field_tests {
    use super::{aspect_from, color_from, h264_profile_level, Durations};
    use ffrwd_wasm::nut;
    use ffrwd_wasm_runtime::runtime;

    fn video_media(colorspace_type: u64) -> nut::Media {
        nut::Media::Video {
            width: 16,
            height: 16,
            sample_width: 1,
            sample_height: 1,
            colorspace_type,
        }
    }

    #[test]
    fn the_nut_colorspace_codes_read_as_ffmpeg_names() {
        for (coded, range, space) in [
            (1u64, "tv", "bt470bg"),
            (2, "tv", "bt709"),
            (17, "pc", "bt470bg"),
            (18, "pc", "bt709"),
        ] {
            let color = color_from(&video_media(coded)).expect("a named colorspace");
            assert_eq!((color.range, color.space), (range, space), "code {coded}");
            assert_eq!(color.primaries, "unknown", "the wire does not say");
            assert_eq!(color.trc, "unknown", "the wire does not say");
        }
    }

    #[test]
    fn an_unknown_or_unnamed_colorspace_carries_nothing() {
        for coded in [0u64, 3, 16, 40] {
            assert_eq!(color_from(&video_media(coded)), None, "code {coded}");
        }
        let audio = nut::Media::Audio {
            sample_rate: 48000,
            channels: 2,
        };
        assert_eq!(color_from(&audio), None);
    }

    #[test]
    fn the_aspect_ratio_is_the_headers_or_nothing() {
        assert_eq!(aspect_from(4, 3), Some((4, 3)));
        assert_eq!(aspect_from(1, 1), Some((1, 1)));
        // 0/0 is the wire's unknown, and a half-zero pair is no ratio.
        assert_eq!(aspect_from(0, 0), None);
        assert_eq!(aspect_from(4, 0), None);
        assert_eq!(aspect_from(u64::MAX, 1), None);
    }

    #[test]
    fn the_h264_profile_and_level_come_off_the_annex_b_sps() {
        // SPS behind a 4-byte start code: profile 0x64 (High), level 0x1f,
        // then the PPS the extradata also carries.
        let extradata = [
            0u8, 0, 0, 1, 0x67, 0x64, 0x00, 0x1f, 0xab, // SPS
            0, 0, 0, 1, 0x68, 0xee, // PPS
        ];
        assert_eq!(h264_profile_level(&extradata), Some((0x64, 0x1f)));
        // A PPS first does not confuse the scan.
        let pps_first = [0u8, 0, 0, 1, 0x68, 0xee, 0, 0, 1, 0x67, 0x42, 0xc0, 0x1e];
        assert_eq!(h264_profile_level(&pps_first), Some((0x42, 0x1e)));
    }

    #[test]
    fn avcc_shaped_extradata_reads_the_same_bytes_off_its_header() {
        assert_eq!(
            h264_profile_level(&[1, 0x64, 0x00, 0x1f, 0xff, 0xe1]),
            Some((0x64, 0x1f))
        );
    }

    #[test]
    fn extradata_without_a_readable_sps_names_no_profile() {
        assert_eq!(h264_profile_level(&[]), None);
        assert_eq!(h264_profile_level(&[0, 0, 0, 1, 0x68, 0xee]), None);
        assert_eq!(h264_profile_level(&[0xab, 0xcd, 0xef]), None);
    }

    fn packet(pts: i64) -> runtime::Packet {
        runtime::Packet {
            pts,
            dts: Some(pts),
            duration: None,
            keyframe: true,
            data: vec![0u8; 4],
        }
    }

    #[test]
    fn the_next_pts_settles_the_previous_packets_duration() {
        let mut durations = Durations::new(0);
        assert!(
            durations.push(packet(0)).is_none(),
            "held for its successor"
        );
        let first = durations.push(packet(1024)).expect("settled");
        assert_eq!((first.pts, first.duration), (0, Some(1024)));
        let second = durations.push(packet(2048)).expect("settled");
        assert_eq!((second.pts, second.duration), (1024, Some(1024)));
        // The final packet has no successor, so its duration stays unknown.
        let last = durations.finish().expect("the held tail");
        assert_eq!((last.pts, last.duration), (2048, None));
        assert!(durations.finish().is_none());
    }

    #[test]
    fn a_reordering_stream_settles_no_durations_and_holds_nothing() {
        let mut durations = Durations::new(2);
        let through = durations.push(packet(7)).expect("straight through");
        assert_eq!((through.pts, through.duration), (7, None));
        assert!(durations.finish().is_none(), "nothing was held");
    }

    #[test]
    fn a_step_that_is_not_forward_settles_nothing() {
        let mut durations = Durations::new(0);
        assert!(durations.push(packet(5)).is_none());
        let first = durations.push(packet(5)).expect("released");
        assert_eq!(first.duration, None, "unknown is never 0");
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

#[cfg(test)]
mod pad_spec_tests {
    use super::{resolve_pad, PadRendition, PadSpec};

    #[test]
    fn a_pad_with_no_spec_is_its_own_index_and_no_rendition() {
        let pads: Vec<Option<PadSpec>> = vec![None, None];
        assert_eq!(resolve_pad(&pads, 0), (0, PadRendition::default()));
        assert_eq!(resolve_pad(&pads, 1), (1, PadRendition::default()));
    }

    #[test]
    fn a_pad_past_the_end_of_the_list_is_its_own_index_too() {
        let pads: Vec<Option<PadSpec>> = vec![None];
        assert_eq!(resolve_pad(&pads, 5), (5, PadRendition::default()));
    }

    #[test]
    fn an_empty_pad_object_still_defaults_to_its_own_index() {
        let spec: PadSpec = serde_json::from_str("{}").expect("valid, if empty");
        let pads = vec![Some(spec)];
        assert_eq!(resolve_pad(&pads, 3), (3, PadRendition::default()));
    }

    #[test]
    fn a_row_overrides_the_index_and_leaves_rendition_default() {
        let spec: PadSpec = serde_json::from_str(r#"{"row":7}"#).expect("valid");
        let pads = vec![Some(spec)];
        assert_eq!(resolve_pad(&pads, 0), (7, PadRendition::default()));
    }

    #[test]
    fn a_rendition_with_some_fields_leaves_the_rest_none() {
        let spec: PadSpec =
            serde_json::from_str(r#"{"row":2,"rendition":{"name":"720p","bandwidth":2500000}}"#)
                .expect("valid");
        let pads = vec![Some(spec)];
        let (row, rendition) = resolve_pad(&pads, 0);
        assert_eq!(row, 2);
        assert_eq!(rendition.name.as_deref(), Some("720p"));
        assert_eq!(rendition.bandwidth, Some(2_500_000));
        assert_eq!(rendition.codecs, None);
        assert_eq!(rendition.language, None);
    }

    #[test]
    fn every_rendition_field_round_trips() {
        let spec: PadSpec = serde_json::from_str(
            r#"{"row":1,"rendition":{"name":"720p","bandwidth":1,"codecs":"avc1.640028","language":"en"}}"#,
        )
        .expect("valid");
        assert_eq!(
            spec,
            PadSpec {
                row: Some(1),
                rendition: PadRendition {
                    name: Some("720p".to_string()),
                    bandwidth: Some(1),
                    codecs: Some("avc1.640028".to_string()),
                    language: Some("en".to_string()),
                },
            }
        );
    }

    #[test]
    fn malformed_pad_json_is_rejected() {
        let err = serde_json::from_str::<PadSpec>("not json").unwrap_err();
        assert!(!err.to_string().is_empty(), "serde names the parse failure");
    }
}

#[cfg(test)]
mod coded_stream_for_tests {
    use super::{coded_stream_for, colorspace_type_for};
    use ffrwd_wasm_runtime::runtime::{CodedFormat, CodedStream, ColorInfo, TimeBase};

    fn h264() -> CodedStream {
        CodedStream {
            codec: "h264".to_string(),
            time_base: TimeBase { num: 1, den: 25 },
            format: CodedFormat::Video {
                width: 64,
                height: 48,
                sample_aspect_ratio: None,
                color: None,
            },
            extradata: vec![0x67, 0x42, 0x00, 0x1e],
            profile: Some(0x42),
            level: Some(0x1e),
        }
    }

    #[test]
    fn a_video_codec_gets_the_muxers_own_fourcc_and_the_settled_decode_delay() {
        let stream = coded_stream_for(&h264(), 2).expect("h264 is carried");
        assert_eq!(stream.fourcc, b"H264");
        assert_eq!(stream.decode_delay, 2);
        assert_eq!(stream.extradata, h264().extradata);
        assert_eq!(
            stream.time_base,
            ffrwd_wasm::nut::TimeBase { num: 1, den: 25 }
        );
        assert_eq!(
            stream.media,
            ffrwd_wasm::nut::Media::Video {
                width: 64,
                height: 48,
                // No sample_aspect_ratio: the unset default matches
                // `Stream::video`'s own.
                sample_width: 1,
                sample_height: 1,
                colorspace_type: 0,
            }
        );
    }

    #[test]
    fn an_audio_codec_carries_its_rate_and_channel_count() {
        let coded = CodedStream {
            codec: "aac".to_string(),
            time_base: TimeBase { num: 1, den: 48000 },
            format: CodedFormat::Audio {
                sample_rate: 48000,
                channels: 2,
                channel_layout: Some("stereo"),
            },
            extradata: vec![0x11, 0x90],
            profile: None,
            level: None,
        };
        let stream = coded_stream_for(&coded, 0).expect("aac is carried");
        assert_eq!(stream.fourcc, b"\xff\x00\x00\x00");
        assert_eq!(stream.decode_delay, 0);
        assert_eq!(
            stream.media,
            ffrwd_wasm::nut::Media::Audio {
                sample_rate: 48000,
                channels: 2,
            }
        );
    }

    #[test]
    fn a_codec_this_wire_does_not_carry_is_named_in_the_error() {
        let mut coded = h264();
        coded.codec = "vp9".to_string();
        let err = coded_stream_for(&coded, 0).unwrap_err().to_string();
        assert!(err.contains("vp9"), "{err}");
        assert!(err.contains("h264"), "{err} should list what IS carried");
    }

    #[test]
    fn the_pixel_aspect_ratio_survives_when_the_source_declares_one() {
        let mut coded = h264();
        let CodedFormat::Video {
            sample_aspect_ratio,
            ..
        } = &mut coded.format
        else {
            unreachable!()
        };
        *sample_aspect_ratio = Some((4, 3));
        let stream = coded_stream_for(&coded, 0).expect("h264 is carried");
        let ffrwd_wasm::nut::Media::Video {
            sample_width,
            sample_height,
            ..
        } = stream.media
        else {
            unreachable!()
        };
        assert_eq!((sample_width, sample_height), (4, 3));
    }

    #[test]
    fn colorspace_type_for_is_color_froms_inverse_on_what_it_can_carry() {
        // The wire only ever writes these four combinations (nut::mod.rs
        // tests pin the same pairs the other way).
        for (code, range, space) in [
            (1u64, "tv", "bt470bg"),
            (2, "tv", "bt709"),
            (17, "pc", "bt470bg"),
            (18, "pc", "bt709"),
        ] {
            let color = ColorInfo {
                range,
                primaries: "unknown",
                trc: "unknown",
                space,
            };
            assert_eq!(colorspace_type_for(Some(&color)), code, "{range} {space}");
        }
    }

    #[test]
    fn no_color_and_an_unnamed_matrix_both_carry_nothing() {
        assert_eq!(colorspace_type_for(None), 0);
        let color = ColorInfo {
            range: "tv",
            primaries: "unknown",
            trc: "unknown",
            space: "ycgco",
        };
        assert_eq!(
            colorspace_type_for(Some(&color)),
            0,
            "no code for this matrix"
        );
    }
}

#[cfg(test)]
mod catalog_json_tests {
    use super::catalog_json;
    use ffrwd_wasm_runtime::runtime::{
        Catalog, CodedFormat, CodedStream, RenditionMeta, SourceTrack, StreamInfo, TimeBase,
    };

    fn track(row: u32) -> SourceTrack {
        SourceTrack {
            stream: CodedStream {
                codec: "h264".to_string(),
                time_base: TimeBase { num: 1, den: 25 },
                format: CodedFormat::Video {
                    width: 64,
                    height: 48,
                    sample_aspect_ratio: None,
                    color: None,
                },
                extradata: vec![0xde, 0xad, 0xbe, 0xef],
                profile: Some(66),
                level: Some(30),
            },
            info: StreamInfo {
                index: row,
                kind: "video".to_string(),
                codec: "h264".to_string(),
                duration: None,
                tags: Vec::new(),
            },
            row,
            rendition: RenditionMeta::default(),
        }
    }

    #[test]
    fn a_bounded_single_track_catalog_prints_the_documented_shape() {
        let catalog = Catalog {
            tracks: vec![track(0)],
            bounded: true,
        };
        let json = serde_json::to_value(catalog_json(&catalog)).expect("serializes");
        assert_eq!(json["bounded"], true);
        let track = &json["tracks"][0];
        assert_eq!(track["codec"], "h264");
        assert_eq!(track["time_base"], serde_json::json!([1, 25]));
        assert_eq!(
            track["format"],
            serde_json::json!({"video": {"width": 64, "height": 48}})
        );
        assert_eq!(track["extradata"], "deadbeef");
        assert_eq!(track["profile"], 66);
        assert_eq!(track["level"], 30);
        assert_eq!(track["row"], 0);
        assert_eq!(
            track["rendition"],
            serde_json::json!({"name": null, "bandwidth": null, "codecs": null, "language": null})
        );
    }

    #[test]
    fn an_audio_track_nests_rate_and_channels_instead_of_geometry() {
        let mut audio = track(1);
        audio.stream.format = CodedFormat::Audio {
            sample_rate: 48000,
            channels: 2,
            channel_layout: None,
        };
        audio.stream.codec = "aac".to_string();
        audio.rendition = RenditionMeta {
            name: Some("audio".to_string()),
            bandwidth: Some(128_000),
            codecs: Some("mp4a.40.2".to_string()),
            language: Some("en".to_string()),
        };
        let catalog = Catalog {
            tracks: vec![audio],
            bounded: false,
        };
        let json = serde_json::to_value(catalog_json(&catalog)).expect("serializes");
        assert_eq!(json["bounded"], false);
        let track = &json["tracks"][0];
        assert_eq!(
            track["format"],
            serde_json::json!({"audio": {"sample_rate": 48000, "channels": 2}})
        );
        assert_eq!(
            track["rendition"],
            serde_json::json!({
                "name": "audio", "bandwidth": 128000, "codecs": "mp4a.40.2", "language": "en"
            })
        );
    }
}

#[cfg(test)]
mod grant_args_tests {
    use super::take_grant_args;

    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| s.to_string()).collect()
    }

    /// `-http` ahead of `--invoke` is taken out before dispatch, the same
    /// way it is ahead of a run, so a compile-time invoke of an
    /// http-importing module can be granted.
    #[test]
    fn a_leading_http_grant_is_taken_out_ahead_of_invoke() {
        let argv = strings(&[
            "-http",
            "module.wasm",
            "--invoke",
            "module.wasm",
            "fn",
            "{}",
        ]);
        let rest = take_grant_args(argv).expect("parses");
        assert_eq!(rest, strings(&["--invoke", "module.wasm", "fn", "{}"]));
    }

    /// `-net` is taken out the same way, for the udp grant's symmetry with
    /// the http one.
    #[test]
    fn a_net_grant_is_taken_out_the_same_way() {
        let argv = strings(&["-net", "module.wasm", "--invoke", "module.wasm", "fn", "{}"]);
        let rest = take_grant_args(argv).expect("parses");
        assert_eq!(rest, strings(&["--invoke", "module.wasm", "fn", "{}"]));
    }

    #[test]
    fn a_grant_with_no_value_is_refused() {
        let err = take_grant_args(strings(&["-http"])).unwrap_err();
        assert_eq!(err.to_string(), "-http requires a value");
    }

    #[test]
    fn argv_carrying_no_grant_passes_through_unchanged() {
        let argv = strings(&["--invoke", "module.wasm", "fn", "{}"]);
        let rest = take_grant_args(argv.clone()).expect("parses");
        assert_eq!(rest, argv);
    }
}
