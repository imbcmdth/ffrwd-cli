//! wasmtime wrapper: compiles, instantiates and drives a `ffrwd:av` filter
//! component. Nothing above this module sees a wasmtime type.
//!
//! Every module is driven through one windowed call, whatever it exports and
//! whatever wit world it was built against. A module exporting the current
//! world's `window-filter` gets its own `process`; everything older is adapted
//! and never learns the difference. The adapters are what let a module built
//! years apart from this host still load: see [`WORLDS`].
//!
//! A module is one kind, which its own description says: it names pixel
//! formats or it names sample formats. Only a windowed interface from 0.7.0
//! on can be opened for audio at all, so an audio stream reaching anything
//! older is refused naming the module and its world.
//!
//! A module says how many streams it reads. Only the windowed interface from
//! 0.9.0 on has the field, so every older world is adapted as reading one.
//!
//! A packet sink is the one export outside the windowed call: encoded packets
//! in, rows out, nothing adapted because only the current world has the
//! interface. It is hosted by [`PacketSink`], beside [`Filter`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use anyhow::{anyhow, bail, Context, Result};
use wasmtime::component::{Component, Linker, ResourceTable};
use wasmtime::{Engine, Store};
use wasmtime_wasi::{WasiCtx, WasiCtxView, WasiView};
use wasmtime_wasi_http::{WasiHttpCtx, WasiHttpCtxView, WasiHttpView};
use wasmtime_wasi_nn::wit::{WasiNnCtx, WasiNnView};

use crate::egress::{self, NetPolicy};
use crate::nn;

/// The wit package versions this host loads, newest first. A component is
/// matched against them in that order, so a module built against the current
/// world is recognised without consulting the older ones. The current world is
/// `wit/`; each older one is kept whole under `worlds/<version>/`.
pub const WORLDS: &[&str] = &[
    "0.10.0", "0.9.0", "0.8.0", "0.7.0", "0.6.0", "0.5.0", "0.4.0", "0.3.0", "0.2.0",
];

/// The world new modules target, and the one every older world is adapted to.
pub const WORLD: &str = "0.10.0";

/// The wit package a module targets, spelled as it appears in a module
/// description.
pub const WORLD_PACKAGE: &str = "ffrwd:av@0.10.0";

/// The component export name an interface carries in a given world.
fn interface(name: &str, world: &str) -> String {
    format!("ffrwd:av/{name}@{world}")
}

// One bindgen expansion per world per world-shape. The generated types are
// distinct per package version even where the wit text is identical, so the
// adapters below convert rather than share.

/// The module description of a world that names pixel formats alone. Every
/// world before 0.7.0 is one, so their modules are video by construction.
macro_rules! video_only_meta {
    () => {
        pub fn meta(m: video::ffrwd::av::types::Meta) -> crate::runtime::Meta {
            crate::runtime::Meta {
                name: m.name,
                version: m.version,
                params_schema: m.params_schema,
                rows_schema: m.rows_schema,
                pixel_formats: m.pixel_formats,
                sample_formats: Vec::new(),
                sample_rates: Vec::new(),
                channel_counts: Vec::new(),
                rows_language: Vec::new(),
            }
        }
    };
}

/// The stream info a world with a time-base field hands a module: nothing is
/// stamped onto the tags.
macro_rules! stream_info_with_time_base {
    () => {
        pub fn stream_info(
            stream: &crate::runtime::StreamInfo,
            time_base: crate::runtime::TimeBase,
            name: &str,
        ) -> anyhow::Result<video::ffrwd::av::types::StreamInfo> {
            let (num, den) = time_base.rational(name)?;
            Ok(video::ffrwd::av::types::StreamInfo {
                index: stream.index,
                kind: stream.kind.clone(),
                codec: stream.codec.clone(),
                duration: stream.duration,
                tags: stream.tags.clone(),
                time_base: video::ffrwd::av::types::Rational { num, den },
            })
        }
    };
}

/// What a kind-bearing world's `init` is handed, from the host's own format.
macro_rules! kind_bearing_format {
    () => {
        pub fn format(format: &crate::runtime::Format) -> video::ffrwd::av::types::Format {
            use video::ffrwd::av::types::{AudioFormat, Format, VideoFormat};
            match format.media {
                crate::runtime::Media::Video(video) => Format::Video(VideoFormat {
                    width: video.width,
                    height: video.height,
                    pix_fmt: video.pix_fmt.to_string(),
                }),
                crate::runtime::Media::Audio(audio) => Format::Audio(AudioFormat {
                    sample_rate: audio.sample_rate,
                    channels: audio.channels,
                    sample_fmt: audio.sample_fmt.to_string(),
                }),
            }
        }
    };
}

/// The module description of every world from 0.8.0 on, which is where a
/// module first named which params settle the language of its rows.
macro_rules! meta_with_rows_language {
    () => {
        pub fn meta(m: video::ffrwd::av::types::Meta) -> crate::runtime::Meta {
            crate::runtime::Meta {
                name: m.name,
                version: m.version,
                params_schema: m.params_schema,
                rows_schema: m.rows_schema,
                pixel_formats: m.pixel_formats,
                sample_formats: m.sample_formats,
                sample_rates: m.sample_rates,
                channel_counts: m.channel_counts,
                rows_language: m.rows_language,
            }
        }
    };
}

mod world_0100 {
    stream_info_with_time_base!();
    kind_bearing_format!();
    meta_with_rows_language!();

    pub mod video {
        wasmtime::component::bindgen!({ path: "../wit", world: "video-module" });
    }
    pub mod meta {
        wasmtime::component::bindgen!({
            path: "../wit",
            world: "meta-module",
            with: {
                "ffrwd:av/types": crate::runtime::world_0100::video::ffrwd::av::types,
                "ffrwd:av/filter": crate::runtime::world_0100::video::exports::ffrwd::av::filter,
            },
        });
    }
    pub mod window {
        wasmtime::component::bindgen!({
            path: "../wit",
            world: "window-module",
            with: { "ffrwd:av/types": crate::runtime::world_0100::video::ffrwd::av::types },
        });
    }
    pub mod packet {
        wasmtime::component::bindgen!({
            path: "../wit",
            world: "packet-module",
            with: { "ffrwd:av/types": crate::runtime::world_0100::video::ffrwd::av::types },
        });
    }
    pub mod values {
        wasmtime::component::bindgen!({ path: "../wit", world: "values-module" });
    }
}

mod world_090 {
    stream_info_with_time_base!();
    kind_bearing_format!();
    meta_with_rows_language!();

    pub mod video {
        wasmtime::component::bindgen!({ path: "../worlds/0.9.0", world: "video-module" });
    }
    pub mod meta {
        wasmtime::component::bindgen!({
            path: "../worlds/0.9.0",
            world: "meta-module",
            with: {
                "ffrwd:av/types": crate::runtime::world_090::video::ffrwd::av::types,
                "ffrwd:av/filter": crate::runtime::world_090::video::exports::ffrwd::av::filter,
            },
        });
    }
    pub mod window {
        wasmtime::component::bindgen!({
            path: "../worlds/0.9.0",
            world: "window-module",
            with: { "ffrwd:av/types": crate::runtime::world_090::video::ffrwd::av::types },
        });
    }
    pub mod values {
        wasmtime::component::bindgen!({ path: "../worlds/0.9.0", world: "values-module" });
    }
}

mod world_080 {
    stream_info_with_time_base!();
    kind_bearing_format!();
    meta_with_rows_language!();

    pub mod video {
        wasmtime::component::bindgen!({ path: "../worlds/0.8.0", world: "video-module" });
    }
    pub mod meta {
        wasmtime::component::bindgen!({
            path: "../worlds/0.8.0",
            world: "meta-module",
            with: {
                "ffrwd:av/types": crate::runtime::world_080::video::ffrwd::av::types,
                "ffrwd:av/filter": crate::runtime::world_080::video::exports::ffrwd::av::filter,
            },
        });
    }
    pub mod window {
        wasmtime::component::bindgen!({
            path: "../worlds/0.8.0",
            world: "window-module",
            with: { "ffrwd:av/types": crate::runtime::world_080::video::ffrwd::av::types },
        });
    }
    pub mod values {
        wasmtime::component::bindgen!({ path: "../worlds/0.8.0", world: "values-module" });
    }
}

mod world_070 {
    stream_info_with_time_base!();
    kind_bearing_format!();

    pub mod video {
        wasmtime::component::bindgen!({ path: "../worlds/0.7.0", world: "video-module" });
    }
    pub mod meta {
        wasmtime::component::bindgen!({
            path: "../worlds/0.7.0",
            world: "meta-module",
            with: {
                "ffrwd:av/types": crate::runtime::world_070::video::ffrwd::av::types,
                "ffrwd:av/filter": crate::runtime::world_070::video::exports::ffrwd::av::filter,
            },
        });
    }
    pub mod window {
        wasmtime::component::bindgen!({
            path: "../worlds/0.7.0",
            world: "window-module",
            with: { "ffrwd:av/types": crate::runtime::world_070::video::ffrwd::av::types },
        });
    }
    pub mod values {
        wasmtime::component::bindgen!({ path: "../worlds/0.7.0", world: "values-module" });
    }

    /// This world names both kinds' formats, and declares no language for the
    /// rows it emits - which is what its adapter answers for it.
    pub fn meta(m: video::ffrwd::av::types::Meta) -> crate::runtime::Meta {
        crate::runtime::Meta {
            name: m.name,
            version: m.version,
            params_schema: m.params_schema,
            rows_schema: m.rows_schema,
            pixel_formats: m.pixel_formats,
            sample_formats: m.sample_formats,
            sample_rates: m.sample_rates,
            channel_counts: m.channel_counts,
            rows_language: Vec::new(),
        }
    }
}

mod world_060 {
    video_only_meta!();
    // This world reads the time base as a field too, and hosts video alone.
    stream_info_with_time_base!();

    pub mod video {
        wasmtime::component::bindgen!({ path: "../worlds/0.6.0", world: "video-module" });
    }
    pub mod meta {
        wasmtime::component::bindgen!({
            path: "../worlds/0.6.0",
            world: "meta-module",
            with: {
                "ffrwd:av/types": crate::runtime::world_060::video::ffrwd::av::types,
                "ffrwd:av/filter": crate::runtime::world_060::video::exports::ffrwd::av::filter,
            },
        });
    }
    pub mod window {
        wasmtime::component::bindgen!({
            path: "../worlds/0.6.0",
            world: "window-module",
            with: { "ffrwd:av/types": crate::runtime::world_060::video::ffrwd::av::types },
        });
    }
    pub mod values {
        wasmtime::component::bindgen!({ path: "../worlds/0.6.0", world: "values-module" });
    }
}

mod world_050 {
    video_only_meta!();

    pub mod video {
        wasmtime::component::bindgen!({ path: "../worlds/0.5.0", world: "video-module" });
    }
    pub mod meta {
        wasmtime::component::bindgen!({
            path: "../worlds/0.5.0",
            world: "meta-module",
            with: {
                "ffrwd:av/types": crate::runtime::world_050::video::ffrwd::av::types,
                "ffrwd:av/filter": crate::runtime::world_050::video::exports::ffrwd::av::filter,
            },
        });
    }
    pub mod window {
        wasmtime::component::bindgen!({
            path: "../worlds/0.5.0",
            world: "window-module",
            with: { "ffrwd:av/types": crate::runtime::world_050::video::ffrwd::av::types },
        });
    }
    pub mod values {
        wasmtime::component::bindgen!({ path: "../worlds/0.5.0", world: "values-module" });
    }

    /// This world has no time-base field, so the host stamps the tag its
    /// modules read instead.
    pub fn stream_info(
        stream: &crate::runtime::StreamInfo,
        time_base: crate::runtime::TimeBase,
        _name: &str,
    ) -> anyhow::Result<video::ffrwd::av::types::StreamInfo> {
        Ok(video::ffrwd::av::types::StreamInfo {
            index: stream.index,
            kind: stream.kind.clone(),
            codec: stream.codec.clone(),
            duration: stream.duration,
            tags: crate::runtime::tags_with_time_base(stream, time_base),
        })
    }
}

mod world_040 {
    video_only_meta!();

    pub mod video {
        wasmtime::component::bindgen!({ path: "../worlds/0.4.0", world: "video-module" });
    }
    pub mod meta {
        wasmtime::component::bindgen!({
            path: "../worlds/0.4.0",
            world: "meta-module",
            with: {
                "ffrwd:av/types": crate::runtime::world_040::video::ffrwd::av::types,
                "ffrwd:av/filter": crate::runtime::world_040::video::exports::ffrwd::av::filter,
            },
        });
    }
    pub mod values {
        wasmtime::component::bindgen!({ path: "../worlds/0.4.0", world: "values-module" });
    }

    pub fn stream_info(
        stream: &crate::runtime::StreamInfo,
        time_base: crate::runtime::TimeBase,
        _name: &str,
    ) -> anyhow::Result<video::ffrwd::av::types::StreamInfo> {
        Ok(video::ffrwd::av::types::StreamInfo {
            index: stream.index,
            kind: stream.kind.clone(),
            codec: stream.codec.clone(),
            duration: stream.duration,
            tags: crate::runtime::tags_with_time_base(stream, time_base),
        })
    }
}

mod world_030 {
    video_only_meta!();

    pub mod video {
        wasmtime::component::bindgen!({ path: "../worlds/0.3.0", world: "video-module" });
    }
    pub mod values {
        wasmtime::component::bindgen!({ path: "../worlds/0.3.0", world: "values-module" });
    }

    pub fn stream_info(
        stream: &crate::runtime::StreamInfo,
        time_base: crate::runtime::TimeBase,
        _name: &str,
    ) -> anyhow::Result<video::ffrwd::av::types::StreamInfo> {
        Ok(video::ffrwd::av::types::StreamInfo {
            index: stream.index,
            kind: stream.kind.clone(),
            codec: stream.codec.clone(),
            duration: stream.duration,
            tags: crate::runtime::tags_with_time_base(stream, time_base),
        })
    }
}

mod world_020 {
    video_only_meta!();

    pub mod video {
        wasmtime::component::bindgen!({ path: "../worlds/0.2.0", world: "video-module" });
    }

    pub fn stream_info(
        stream: &crate::runtime::StreamInfo,
        time_base: crate::runtime::TimeBase,
        _name: &str,
    ) -> anyhow::Result<video::ffrwd::av::types::StreamInfo> {
        Ok(video::ffrwd::av::types::StreamInfo {
            index: stream.index,
            kind: stream.kind.clone(),
            codec: stream.codec.clone(),
            duration: stream.duration,
            tags: crate::runtime::tags_with_time_base(stream, time_base),
        })
    }
}

/// wasmtime is built without its `anyhow` feature, so `wasmtime::Error` does
/// not convert on its own.
fn wasm_err(e: wasmtime::Error) -> anyhow::Error {
    anyhow::Error::from_boxed(e.into_boxed_dyn_error())
}

/// The stream an instance is attached to, told to the module once at init.
#[derive(Debug, Clone, Default)]
pub struct StreamInfo {
    pub index: u32,
    pub kind: String,
    pub codec: String,
    pub duration: Option<f64>,
    pub tags: Vec<(String, String)>,
}

/// The unit timestamps are counted in, as a rational number of seconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeBase {
    pub num: u64,
    pub den: u64,
}

impl TimeBase {
    /// A timestamp in seconds. Only an adapter needs it: an older world's
    /// per-frame interface takes a time in seconds where the windowed one
    /// takes the timestamp itself.
    fn seconds(&self, pts: i64) -> f64 {
        pts as f64 * self.num as f64 / self.den as f64
    }

    /// `num/den`, the spelling the `time-base` stream tag carries.
    fn tag(&self) -> String {
        format!("{}/{}", self.num, self.den)
    }

    /// The pair the current world hands a module as a field.
    fn rational(&self, name: &str) -> Result<(i32, i32)> {
        let num = i32::try_from(self.num).ok();
        let den = i32::try_from(self.den).ok().filter(|d| *d > 0);
        match (num, den) {
            (Some(num), Some(den)) => Ok((num, den)),
            _ => bail!(
                "{name}: time base {}/{} does not fit the rational a module is handed",
                self.num,
                self.den
            ),
        }
    }
}

/// Stream tag naming the unit timestamps are counted in, for a world whose
/// stream info has no field for it.
const TIME_BASE_TAG: &str = "time-base";

/// The stream's tags with the time base stamped on, replacing whatever the
/// caller carried under that name. The host settles the time base, so its tag
/// is the host's to write.
fn tags_with_time_base(stream: &StreamInfo, time_base: TimeBase) -> Vec<(String, String)> {
    let mut tags: Vec<(String, String)> = stream
        .tags
        .iter()
        .filter(|(key, _)| key != TIME_BASE_TAG)
        .cloned()
        .collect();
    tags.push((TIME_BASE_TAG.to_string(), time_base.tag()));
    tags
}

/// The frames a video instance is opened for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoFormat {
    pub width: u32,
    pub height: u32,
    pub pix_fmt: &'static str,
    /// Byte size of one frame in `pix_fmt`, which only the caller can work
    /// out.
    pub frame_len: usize,
}

/// The samples an audio instance is opened for. One sample is one instant
/// across every channel, which is the unit windows and strides count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioFormat {
    pub sample_rate: u32,
    pub channels: u32,
    /// `f32` or `s16`, interleaved.
    pub sample_fmt: &'static str,
}

impl AudioFormat {
    /// Bytes one sample occupies, every channel included.
    pub fn sample_len(&self) -> usize {
        let width = match self.sample_fmt {
            "s16" => 2,
            _ => 4,
        };
        width * self.channels as usize
    }
}

/// Which kind of stream an instance carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Media {
    Video(VideoFormat),
    Audio(AudioFormat),
}

/// Everything about the payloads an instance is opened for, fixed for its
/// life.
#[derive(Debug, Clone, Copy)]
pub struct Format {
    pub media: Media,
    pub time_base: TimeBase,
}

impl Format {
    /// The video geometry, or None for an audio instance.
    pub fn video(&self) -> Option<VideoFormat> {
        match self.media {
            Media::Video(video) => Some(video),
            Media::Audio(_) => None,
        }
    }

    /// The audio geometry, or None for a video instance.
    pub fn audio(&self) -> Option<AudioFormat> {
        match self.media {
            Media::Audio(audio) => Some(audio),
            Media::Video(_) => None,
        }
    }

    /// `video` or `audio`, for a message.
    pub fn kind(&self) -> Kind {
        match self.media {
            Media::Video(_) => Kind::Video,
            Media::Audio(_) => Kind::Audio,
        }
    }
}

/// The two stream kinds a module can be, as its declaration says and as a
/// stream arrives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Video,
    Audio,
}

impl std::fmt::Display for Kind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Kind::Video => "video",
            Kind::Audio => "audio",
        })
    }
}

/// Static description of a module, from its `describe()`. A module fills in
/// the formats of one kind and leaves the other kind's empty.
#[derive(Debug, Clone)]
pub struct Meta {
    pub name: String,
    pub version: String,
    pub params_schema: String,
    pub rows_schema: String,
    pub pixel_formats: Vec<String>,
    pub sample_formats: Vec<String>,
    /// Sample rates accepted; empty is every rate.
    pub sample_rates: Vec<u32>,
    /// Channel counts accepted; empty is every count.
    pub channel_counts: Vec<u32>,
    /// Ordered param names; the rows' language is the first of these params
    /// that is set at the call. Empty when the module declares none, which is
    /// what every world before 0.8.0 answers.
    pub rows_language: Vec<String>,
}

impl Meta {
    /// The kind this module declares itself, or an error naming the module
    /// when it declares both kinds or neither.
    pub fn kind(&self) -> Result<Kind> {
        match (
            self.pixel_formats.is_empty(),
            self.sample_formats.is_empty(),
        ) {
            (false, true) => Ok(Kind::Video),
            (true, false) => Ok(Kind::Audio),
            (false, false) => bail!(
                "{} publishes pixel formats ({}) and sample formats ({}); a module is one kind",
                self.name,
                self.pixel_formats.join(", "),
                self.sample_formats.join(", ")
            ),
            (true, true) => bail!(
                "{} publishes neither pixel formats nor sample formats, so it is neither a video \
                 module nor an audio one",
                self.name
            ),
        }
    }
}

/// How the host must drive a module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    /// Frames one call receives.
    pub window: u32,
    /// Frames consumed per call.
    pub stride: u32,
    /// Whether a call depends only on the frames it was handed.
    pub pure: bool,
    /// Whether every call returns one output per frame it consumed, at that
    /// frame's own timestamp.
    pub one_to_one: bool,
}

/// What a module exporting an older world's per-frame interface is driven as.
const ADAPTED_SHAPE: Shape = Shape {
    window: 1,
    stride: 1,
    pure: false,
    one_to_one: true,
};

/// Static description of one value function, from `list-functions()`.
#[derive(Debug, Clone)]
pub struct FunctionMeta {
    pub name: String,
    pub params_schema: String,
    pub result_schema: String,
}

/// One frame on the wire between the host and a module, or between two
/// modules: its timestamp, its pixels, and the rows travelling with it.
#[derive(Debug, Clone)]
pub struct Frame {
    pub pts: i64,
    pub data: Vec<u8>,
    pub rows: Vec<String>,
}

/// What one call produced: frames, and the rows that had no frame to ride.
#[derive(Debug, Clone, Default)]
pub struct Processed {
    pub frames: Vec<Frame>,
    pub trailing: Vec<String>,
}

/// The encoded stream a packet sink is opened for, fixed for its life.
#[derive(Debug, Clone)]
pub struct CodedStream {
    /// ffmpeg's name for the codec, e.g. `h264`.
    pub codec: String,
    pub time_base: TimeBase,
    pub width: u32,
    pub height: u32,
    /// The codec's out-of-band header, as the container carried it.
    pub extradata: Vec<u8>,
}

/// One encoded packet, exactly as the encoder emitted it.
#[derive(Debug, Clone)]
pub struct Packet {
    pub pts: i64,
    /// Absent for the first packets of a reordering stream, where the wire
    /// does not settle it.
    pub dts: Option<i64>,
    pub keyframe: bool,
    pub data: Vec<u8>,
}

/// What one packet-sink call produced: rows alone.
#[derive(Debug, Clone, Default)]
pub struct Emitted {
    pub rows: Vec<String>,
    /// Rows with no packet left to prompt them; only the final call may
    /// carry any.
    pub trailing: Vec<String>,
}

/// What a packet sink publishes without being opened for any stream.
#[derive(Debug, Clone)]
pub struct DescribedPacketSink {
    pub meta: Meta,
    /// ffmpeg codec names accepted, most preferred first; empty is every
    /// codec.
    pub codecs: Vec<String>,
    /// The wit package version the module was built against.
    pub world: &'static str,
}

/// What a module publishes without being opened for any particular stream.
/// `shape` is absent for a module exporting a per-frame interface, whose
/// purity is not knowable until it has been init-ed.
#[derive(Debug, Clone)]
pub struct Described {
    pub meta: Meta,
    pub shape: Option<Shape>,
    /// Whether the module acts on the rows arriving with its frames: declared
    /// by a windowed module, and read off the exports of a per-frame one.
    pub reads_rows: bool,
    /// Whether upstream rows may leave on this module's own output frames.
    pub forwards_rows: bool,
    /// The wit package version the module was built against.
    pub world: &'static str,
    /// Whether the interface it exports can be opened for an audio stream at
    /// all. Only the current world's windowed interface can.
    pub audio_capable: bool,
    /// How many streams the module reads at once. Declared from 0.9.0 on;
    /// every older world reads one, which is what its adapter answers.
    pub inputs: u32,
}

/// Store data: WASI, and the models a component that imports `wasi:nn` can
/// name. The nn context is empty for every module that does not.
struct Host {
    wasi: WasiCtx,
    table: ResourceTable,
    nn: WasiNnCtx,
    http: WasiHttpCtx,
    hooks: egress::Hooks,
}

impl WasiView for Host {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.wasi,
            table: &mut self.table,
        }
    }
}

impl WasiHttpView for Host {
    fn http(&mut self) -> WasiHttpCtxView<'_> {
        WasiHttpCtxView {
            hooks: &mut self.hooks,
            table: &mut self.table,
            ctx: &mut self.http,
        }
    }
}

fn nn_view(host: &mut Host) -> WasiNnView<'_> {
    WasiNnView::new(&mut host.table, &mut host.nn)
}

/// Whether the component asks the host for inference. Read off the component
/// type, so nothing is instantiated to find out.
fn imports_nn(component: &Component) -> bool {
    imports_interface(component, nn::IMPORT_PREFIX)
}

/// The import prefixes the two granted effects answer to.
const HTTP_IMPORT_PREFIX: &str = "wasi:http/";
const SOCKETS_IMPORT_PREFIX: &str = "wasi:sockets/";

/// Whether the component imports any interface under `prefix`. Read off the
/// component type, so nothing is instantiated to find out.
fn imports_interface(component: &Component, prefix: &str) -> bool {
    component
        .component_type()
        .imports(engine())
        .any(|(name, _)| name.starts_with(prefix))
}

/// The effects the argv granted one module. Deny by default: a module the
/// argv never named gets neither.
#[derive(Clone, Copy, Default)]
struct Grants {
    http: bool,
    net: bool,
}

/// Granted effects per module, keyed by canonical path.
fn grant_table() -> &'static Mutex<HashMap<PathBuf, Grants>> {
    static GRANTS: OnceLock<Mutex<HashMap<PathBuf, Grants>>> = OnceLock::new();
    GRANTS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn granted(module_path: &str) -> Result<Grants> {
    let table = grant_table()
        .lock()
        .map_err(|_| anyhow!("grant table poisoned"))?;
    Ok(table
        .get(&canonical(module_path))
        .copied()
        .unwrap_or_default())
}

/// Records one `-http <module>`: the module at `path` may make outbound HTTP
/// requests.
pub fn grant_http(path: &str) -> Result<()> {
    let mut table = grant_table()
        .lock()
        .map_err(|_| anyhow!("grant table poisoned"))?;
    table.entry(canonical(path)).or_default().http = true;
    Ok(())
}

/// Records one `-net <module>`: the module at `path` may open UDP sockets.
pub fn grant_net(path: &str) -> Result<()> {
    let mut table = grant_table()
        .lock()
        .map_err(|_| anyhow!("grant table poisoned"))?;
    table.entry(canonical(path)).or_default().net = true;
    Ok(())
}

/// The store's WASI context: no preopens, no env, no args; stderr passes
/// through. The network is reachable only for a module granted `-net`, and
/// UDP only: inherit_network opens only the address check, and each protocol
/// stays refused until allowed. Under the public policy the address check
/// refuses non-public destinations; binds stay open.
fn wasi_ctx(effects: Grants, policy: NetPolicy) -> WasiCtx {
    let mut builder = WasiCtx::builder();
    builder.inherit_stderr();
    if effects.net {
        match policy {
            NetPolicy::Unrestricted => {
                builder.inherit_network();
            }
            NetPolicy::Public => {
                builder.socket_addr_check(|addr, reason| {
                    Box::pin(async move { egress::check_socket_addr(addr, reason) })
                });
            }
        }
        builder.allow_udp(true);
    }
    builder.build()
}

/// Why a component is being instantiated. Reading what it publishes needs no
/// model: `describe()` and `list-functions()` touch no graph, and the compiler
/// asks them what a module needs before it can bind anything.
#[derive(Clone, Copy, PartialEq)]
enum Purpose {
    Describe,
    Run,
}

/// Adds WASI to a linker, and wasi-nn and wasi:http as well when the
/// component asks for them. A module that does not import an interface gets
/// an unchanged host.
///
/// A run is where the grants are enforced: a module importing `wasi:http`
/// without its `-http` is refused here, before instantiation, and one
/// importing `wasi:sockets` without its `-net` is refused by the store's
/// WASI context at socket creation. A describe touches no effect and is
/// linked regardless.
fn link(
    component: &Component,
    module_path: &str,
    purpose: Purpose,
) -> Result<(Linker<Host>, WasiNnCtx)> {
    let wants_nn = imports_nn(component);
    if wants_nn && purpose == Purpose::Run {
        nn::require_configured(module_path)?;
    }
    let wants_http = imports_interface(component, HTTP_IMPORT_PREFIX);
    if wants_http && purpose == Purpose::Run && !granted(module_path)?.http {
        bail!(
            "{module_path} imports wasi:http, and this run grants it no network; \
             grant it with -http <module>"
        );
    }

    let mut linker = Linker::new(engine());
    wasmtime_wasi::p2::add_to_linker_sync(&mut linker).map_err(wasm_err)?;
    if wants_nn {
        wasmtime_wasi_nn::wit::add_to_linker(&mut linker, nn_view).map_err(wasm_err)?;
    }
    if wants_http {
        wasmtime_wasi_http::p2::add_only_http_to_linker_sync(&mut linker).map_err(wasm_err)?;
    }

    let nn = if wants_nn {
        nn::store_ctx()
    } else {
        nn::empty_ctx()
    };
    Ok((linker, nn))
}

fn engine() -> &'static Engine {
    static ENGINE: OnceLock<Engine> = OnceLock::new();
    ENGINE.get_or_init(Engine::default)
}

/// Compiled components, keyed by canonical path. Compilation is expensive and
/// `Component` is cheap to clone.
fn component_cache() -> &'static Mutex<HashMap<PathBuf, Component>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, Component>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// A module path's canonical form: the key the component cache and the grant
/// table share, so two spellings of one file agree.
fn canonical(module_path: &str) -> PathBuf {
    let path = Path::new(module_path);
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn compile(module_path: &str) -> Result<Component> {
    let key = canonical(module_path);

    let mut cache = component_cache()
        .lock()
        .map_err(|_| anyhow!("component cache poisoned"))?;
    if let Some(component) = cache.get(&key) {
        return Ok(component.clone());
    }

    let component = Component::from_file(engine(), &key)
        .map_err(wasm_err)
        .with_context(|| format!("loading WASM component {module_path}"))?;
    cache.insert(key, component.clone());
    Ok(component)
}

/// The component's exported interface names. No instantiation needed.
fn component_exports(component: &Component) -> Vec<String> {
    component
        .component_type()
        .exports(engine())
        .map(|(name, _)| name.to_string())
        .collect()
}

/// Whether the component's type declares `interface` among its exports.
fn has_export(component: &Component, interface: &str) -> bool {
    component
        .component_type()
        .get_export(engine(), interface)
        .is_some()
}

/// The newest world in which the component exports `name`.
fn world_exporting(component: &Component, name: &str) -> Option<&'static str> {
    WORLDS
        .iter()
        .copied()
        .find(|world| has_export(component, &interface(name, world)))
}

/// Errors naming the component's actual exports when the values interface is
/// missing from every world. Export names come from the component type, no
/// instantiation needed.
fn check_values_export(component: &Component, module_path: &str) -> Result<()> {
    if world_exporting(component, "values").is_some() {
        return Ok(());
    }
    let wanted = interface("values", WORLD);
    let exports = component_exports(component);
    if exports.is_empty() {
        bail!("{module_path} exports nothing, so not {wanted}");
    }
    bail!(
        "{module_path} does not export {wanted}; it exports {}",
        exports.join(", ")
    );
}

/// A module the host can put frames through exports one of the frame
/// interfaces, in some world this host knows. The refusal names the current
/// world's spellings, and what the component exports instead.
fn check_frame_export(component: &Component, module_path: &str) -> Result<()> {
    if world_exporting(component, "window-filter").is_some()
        || world_exporting(component, "filter").is_some()
    {
        return Ok(());
    }

    let window = interface("window-filter", WORLD);
    let filter = interface("filter", WORLD);
    let exports = component_exports(component);
    if exports.is_empty() {
        bail!("{module_path} exports nothing, so neither {window} nor {filter}");
    }
    bail!(
        "{module_path} exports neither {window} nor {filter}; it exports {}",
        exports.join(", ")
    );
}

/// Whether the component at `module_path` exports a per-frame filter
/// interface, in any world this host knows.
pub fn exports_filter(module_path: &str) -> Result<bool> {
    let component = compile(module_path)?;
    Ok(world_exporting(&component, "filter").is_some())
}

/// Whether the component at `module_path` also exports the meta-filter
/// interface, and so reads the rows an upstream module emitted.
pub fn exports_meta_filter(module_path: &str) -> Result<bool> {
    let component = compile(module_path)?;
    Ok(world_exporting(&component, "meta-filter").is_some())
}

/// Whether the component at `module_path` exports the windowed stream
/// interface, in any world this host knows.
pub fn exports_window_filter(module_path: &str) -> Result<bool> {
    let component = compile(module_path)?;
    Ok(world_exporting(&component, "window-filter").is_some())
}

/// Whether the component at `module_path` exports the values interface.
pub fn exports_values(module_path: &str) -> Result<bool> {
    let component = compile(module_path)?;
    Ok(world_exporting(&component, "values").is_some())
}

/// Whether the component at `module_path` asks the host for inference. Read
/// off its imports, so it is answered without a model bound or a runtime
/// present.
pub fn imports_wasi_nn(module_path: &str) -> Result<bool> {
    let component = compile(module_path)?;
    Ok(imports_nn(&component))
}

/// Whether the component at `module_path` asks the host for outbound HTTP,
/// and so needs an `-http` grant to run at all. Read off its imports.
pub fn imports_wasi_http(module_path: &str) -> Result<bool> {
    let component = compile(module_path)?;
    Ok(imports_interface(&component, HTTP_IMPORT_PREFIX))
}

/// Whether the component at `module_path` asks the host for sockets, and so
/// needs a `-net` grant to reach the network. Read off its imports.
pub fn imports_wasi_sockets(module_path: &str) -> Result<bool> {
    let component = compile(module_path)?;
    Ok(imports_interface(&component, SOCKETS_IMPORT_PREFIX))
}

/// The component's exported interface names, for messages naming what a
/// module actually exports instead of what a caller expected.
pub fn exports(module_path: &str) -> Result<Vec<String>> {
    let component = compile(module_path)?;
    Ok(component_exports(&component))
}

/// One instantiated module, whatever world it was built against, answering
/// the questions the current world asks.
trait Adapter {
    fn described(&self, store: &mut Store<Host>) -> Result<Described>;

    fn init(
        &self,
        store: &mut Store<Host>,
        format: &Format,
        stream: &StreamInfo,
        name: &str,
        params: &str,
    ) -> Result<Result<(), String>>;

    fn set_params(&self, store: &mut Store<Host>, params: &str) -> Result<Result<(), String>>;

    /// Purity, which a per-frame module answers only once init-ed and only at
    /// runtime. None from a windowed module, which publishes it statically.
    fn frame_independent(&self, store: &mut Store<Host>) -> Result<Option<bool>>;

    fn process(
        &self,
        store: &mut Store<Host>,
        format: &Format,
        frames: &[Frame],
        trailing: &[String],
        last: bool,
        same: SameRule<'_>,
    ) -> Result<Processed>;
}

/// Whether a module may say its output is the input it was handed. An audio
/// instance whose windows overlap may not: every sample would leave more than
/// once. The name is the module's, for the refusal.
#[derive(Clone, Copy)]
enum SameRule<'a> {
    Allowed,
    RefusedForOverlap(&'a str),
}

/// A module exporting `filter` alone. One frame per call, the rows it emits
/// its own, and no trailing rows in either direction.
macro_rules! video_adapter {
    ($adapter:ident, $world:ident, $version:literal) => {
        struct $adapter($world::video::VideoModule);

        impl Adapter for $adapter {
            fn described(&self, store: &mut Store<Host>) -> Result<Described> {
                let m = self
                    .0
                    .ffrwd_av_filter()
                    .call_describe(store)
                    .map_err(wasm_err)?;
                Ok(per_frame_described($world::meta(m), false, $version))
            }

            fn init(
                &self,
                store: &mut Store<Host>,
                format: &Format,
                stream: &StreamInfo,
                name: &str,
                params: &str,
            ) -> Result<Result<(), String>> {
                let video = video_only(format, name)?;
                let wit = $world::stream_info(stream, format.time_base, name)?;
                self.0
                    .ffrwd_av_filter()
                    .call_init(
                        store,
                        video.width,
                        video.height,
                        video.pix_fmt,
                        &wit,
                        params,
                    )
                    .map_err(wasm_err)
            }

            fn set_params(
                &self,
                store: &mut Store<Host>,
                params: &str,
            ) -> Result<Result<(), String>> {
                self.0
                    .ffrwd_av_filter()
                    .call_set_params(store, params)
                    .map_err(wasm_err)
            }

            fn frame_independent(&self, store: &mut Store<Host>) -> Result<Option<bool>> {
                self.0
                    .ffrwd_av_filter()
                    .call_frame_independent(store)
                    .map_err(wasm_err)
                    .map(Some)
            }

            fn process(
                &self,
                store: &mut Store<Host>,
                format: &Format,
                frames: &[Frame],
                _trailing: &[String],
                _last: bool,
                _same: SameRule<'_>,
            ) -> Result<Processed> {
                use $world::video::exports::ffrwd::av::filter::{FrameInfo, Output};
                let video = opened_for_video(format);
                let mut out = Vec::with_capacity(frames.len());
                for frame in frames {
                    let info = FrameInfo {
                        width: video.width,
                        height: video.height,
                        time: format.time_base.seconds(frame.pts),
                    };
                    let outcome = self
                        .0
                        .ffrwd_av_filter()
                        .call_process(&mut *store, info, &frame.data)
                        .map_err(wasm_err)?;
                    let data = match outcome.output {
                        Output::Frame(produced) => produced,
                        Output::Passthrough => frame.data.clone(),
                    };
                    out.push(Frame {
                        pts: frame.pts,
                        data,
                        rows: outcome.rows,
                    });
                }
                Ok(Processed {
                    frames: out,
                    trailing: Vec::new(),
                })
            }
        }
    };
}

/// A module exporting `meta-filter` beside `filter`. As above, except that the
/// rows arriving with a frame reach it.
macro_rules! meta_adapter {
    ($adapter:ident, $world:ident, $version:literal) => {
        struct $adapter($world::meta::MetaModule);

        impl Adapter for $adapter {
            fn described(&self, store: &mut Store<Host>) -> Result<Described> {
                let m = self
                    .0
                    .ffrwd_av_filter()
                    .call_describe(store)
                    .map_err(wasm_err)?;
                Ok(per_frame_described($world::meta(m), true, $version))
            }

            fn init(
                &self,
                store: &mut Store<Host>,
                format: &Format,
                stream: &StreamInfo,
                name: &str,
                params: &str,
            ) -> Result<Result<(), String>> {
                let video = video_only(format, name)?;
                let wit = $world::stream_info(stream, format.time_base, name)?;
                self.0
                    .ffrwd_av_filter()
                    .call_init(
                        store,
                        video.width,
                        video.height,
                        video.pix_fmt,
                        &wit,
                        params,
                    )
                    .map_err(wasm_err)
            }

            fn set_params(
                &self,
                store: &mut Store<Host>,
                params: &str,
            ) -> Result<Result<(), String>> {
                self.0
                    .ffrwd_av_filter()
                    .call_set_params(store, params)
                    .map_err(wasm_err)
            }

            fn frame_independent(&self, store: &mut Store<Host>) -> Result<Option<bool>> {
                self.0
                    .ffrwd_av_filter()
                    .call_frame_independent(store)
                    .map_err(wasm_err)
                    .map(Some)
            }

            fn process(
                &self,
                store: &mut Store<Host>,
                format: &Format,
                frames: &[Frame],
                _trailing: &[String],
                _last: bool,
                _same: SameRule<'_>,
            ) -> Result<Processed> {
                use $world::video::exports::ffrwd::av::filter::{FrameInfo, Output};
                let video = opened_for_video(format);
                let mut out = Vec::with_capacity(frames.len());
                for frame in frames {
                    let info = FrameInfo {
                        width: video.width,
                        height: video.height,
                        time: format.time_base.seconds(frame.pts),
                    };
                    let outcome = self
                        .0
                        .ffrwd_av_meta_filter()
                        .call_process_meta(&mut *store, info, &frame.data, &frame.rows)
                        .map_err(wasm_err)?;
                    let data = match outcome.output {
                        Output::Frame(produced) => produced,
                        Output::Passthrough => frame.data.clone(),
                    };
                    out.push(Frame {
                        pts: frame.pts,
                        data,
                        rows: outcome.rows,
                    });
                }
                Ok(Processed {
                    frames: out,
                    trailing: Vec::new(),
                })
            }
        }
    };
}

video_adapter!(Video0100, world_0100, "0.10.0");
video_adapter!(Video090, world_090, "0.9.0");
video_adapter!(Video080, world_080, "0.8.0");
video_adapter!(Video070, world_070, "0.7.0");
video_adapter!(Video060, world_060, "0.6.0");
video_adapter!(Video050, world_050, "0.5.0");
video_adapter!(Video040, world_040, "0.4.0");
video_adapter!(Video030, world_030, "0.3.0");
video_adapter!(Video020, world_020, "0.2.0");

meta_adapter!(Meta0100, world_0100, "0.10.0");
meta_adapter!(Meta090, world_090, "0.9.0");
meta_adapter!(Meta080, world_080, "0.8.0");
meta_adapter!(Meta070, world_070, "0.7.0");
meta_adapter!(Meta060, world_060, "0.6.0");
meta_adapter!(Meta050, world_050, "0.5.0");
meta_adapter!(Meta040, world_040, "0.4.0");

/// What a per-frame module of any world publishes, driven as one window.
/// Rows arriving with a frame stop there, so nothing it emits is an upstream
/// module's. The per-frame interface carries video alone in every world,
/// this one included.
fn per_frame_described(meta: Meta, reads_rows: bool, world: &'static str) -> Described {
    Described {
        meta,
        shape: None,
        reads_rows,
        forwards_rows: false,
        world,
        audio_capable: false,
        inputs: 1,
    }
}

/// The video geometry an instance of a video-only interface is opened for.
/// An audio stream reaching one is refused naming the module.
fn video_only(format: &Format, name: &str) -> Result<VideoFormat> {
    format
        .video()
        .ok_or_else(|| anyhow!("{name} was opened for an audio stream, and it hosts video alone"))
}

/// The video geometry of an instance already opened. `Filter::open` refuses
/// an audio stream before any video-only module is init-ed, so by the time
/// frames are flowing the kind is settled.
fn opened_for_video(format: &Format) -> VideoFormat {
    format
        .video()
        .expect("a video-only module is opened for video alone")
}

/// A windowed module of a world whose `init` is handed a kind-bearing format:
/// its `process` is the one the host is written around, trailing rows and all,
/// and these are the only interfaces an audio stream reaches.
macro_rules! kind_bearing_window_adapter {
    ($adapter:ident, $world:ident, $resolve:ident, $version:literal, $inputs:expr) => {
        struct $adapter($world::window::WindowModule);

        impl Adapter for $adapter {
            fn described(&self, store: &mut Store<Host>) -> Result<Described> {
                let d = self
                    .0
                    .ffrwd_av_window_filter()
                    .call_describe(store)
                    .map_err(wasm_err)?;
                // A world with no field for it reads one stream, which is what
                // the count passed in answers on its behalf.
                let inputs = ($inputs)(&d);
                Ok(Described {
                    meta: $world::meta(d.meta),
                    shape: Some(Shape {
                        window: d.window,
                        stride: d.stride,
                        pure: d.pure,
                        one_to_one: d.one_to_one,
                    }),
                    reads_rows: d.reads_rows,
                    forwards_rows: d.forwards_rows,
                    world: $version,
                    audio_capable: true,
                    inputs,
                })
            }

            fn init(
                &self,
                store: &mut Store<Host>,
                format: &Format,
                stream: &StreamInfo,
                name: &str,
                params: &str,
            ) -> Result<Result<(), String>> {
                let wit = $world::stream_info(stream, format.time_base, name)?;
                self.0
                    .ffrwd_av_window_filter()
                    .call_init(store, &$world::format(format), &wit, params)
                    .map_err(wasm_err)
            }

            fn set_params(
                &self,
                store: &mut Store<Host>,
                params: &str,
            ) -> Result<Result<(), String>> {
                self.0
                    .ffrwd_av_window_filter()
                    .call_set_params(store, params)
                    .map_err(wasm_err)
            }

            fn frame_independent(&self, _store: &mut Store<Host>) -> Result<Option<bool>> {
                Ok(None)
            }

            fn process(
                &self,
                store: &mut Store<Host>,
                _format: &Format,
                frames: &[Frame],
                trailing: &[String],
                last: bool,
                same: SameRule<'_>,
            ) -> Result<Processed> {
                use $world::window::exports::ffrwd::av::window_filter::InFrame;
                let window: Vec<InFrame> = frames
                    .iter()
                    .map(|f| InFrame {
                        pts: f.pts,
                        frame: f.data.clone(),
                        rows: f.rows.clone(),
                    })
                    .collect();
                let produced = self
                    .0
                    .ffrwd_av_window_filter()
                    .call_process(store, &window, trailing, last)
                    .map_err(wasm_err)?;
                Ok(Processed {
                    frames: $resolve(produced.frames, frames, same)?,
                    trailing: produced.trailing,
                })
            }
        }
    };
}

kind_bearing_window_adapter!(
    Window0100,
    world_0100,
    resolve_0100,
    "0.10.0",
    |d: &world_0100::window::exports::ffrwd::av::window_filter::WindowMeta| d.inputs
);
kind_bearing_window_adapter!(
    Window090,
    world_090,
    resolve_090,
    "0.9.0",
    |d: &world_090::window::exports::ffrwd::av::window_filter::WindowMeta| d.inputs
);
kind_bearing_window_adapter!(
    Window080,
    world_080,
    resolve_080,
    "0.8.0",
    |_: &world_080::window::exports::ffrwd::av::window_filter::WindowMeta| 1
);
kind_bearing_window_adapter!(
    Window070,
    world_070,
    resolve_070,
    "0.7.0",
    |_: &world_070::window::exports::ffrwd::av::window_filter::WindowMeta| 1
);

/// A windowed module of the world before the format became kind-bearing. It
/// is opened with a frame size and a pixel format, so it hosts video alone.
struct Window060(world_060::window::WindowModule);

impl Adapter for Window060 {
    fn described(&self, store: &mut Store<Host>) -> Result<Described> {
        let d = self
            .0
            .ffrwd_av_window_filter()
            .call_describe(store)
            .map_err(wasm_err)?;
        Ok(Described {
            meta: world_060::meta(d.meta),
            shape: Some(Shape {
                window: d.window,
                stride: d.stride,
                pure: d.pure,
                one_to_one: d.one_to_one,
            }),
            reads_rows: d.reads_rows,
            forwards_rows: d.forwards_rows,
            world: "0.6.0",
            audio_capable: false,
            inputs: 1,
        })
    }

    fn init(
        &self,
        store: &mut Store<Host>,
        format: &Format,
        stream: &StreamInfo,
        name: &str,
        params: &str,
    ) -> Result<Result<(), String>> {
        let video = video_only(format, name)?;
        let wit = world_060::stream_info(stream, format.time_base, name)?;
        self.0
            .ffrwd_av_window_filter()
            .call_init(
                store,
                video.width,
                video.height,
                video.pix_fmt,
                &wit,
                params,
            )
            .map_err(wasm_err)
    }

    fn set_params(&self, store: &mut Store<Host>, params: &str) -> Result<Result<(), String>> {
        self.0
            .ffrwd_av_window_filter()
            .call_set_params(store, params)
            .map_err(wasm_err)
    }

    fn frame_independent(&self, _store: &mut Store<Host>) -> Result<Option<bool>> {
        Ok(None)
    }

    fn process(
        &self,
        store: &mut Store<Host>,
        _format: &Format,
        frames: &[Frame],
        trailing: &[String],
        last: bool,
        same: SameRule<'_>,
    ) -> Result<Processed> {
        use world_060::window::exports::ffrwd::av::window_filter::InFrame;
        let window: Vec<InFrame> = frames
            .iter()
            .map(|f| InFrame {
                pts: f.pts,
                frame: f.data.clone(),
                rows: f.rows.clone(),
            })
            .collect();
        let produced = self
            .0
            .ffrwd_av_window_filter()
            .call_process(store, &window, trailing, last)
            .map_err(wasm_err)?;
        Ok(Processed {
            frames: resolve_060(produced.frames, frames, same)?,
            trailing: produced.trailing,
        })
    }
}

/// A windowed module of the world before trailing rows and forwarding
/// declarations. It is handed no trailing rows and returns none, and its
/// output frames are assumed to carry whatever arrived on their inputs -
/// which is what the host assumed of every windowed module then.
struct Window050(world_050::window::WindowModule);

impl Adapter for Window050 {
    fn described(&self, store: &mut Store<Host>) -> Result<Described> {
        let d = self
            .0
            .ffrwd_av_window_filter()
            .call_describe(store)
            .map_err(wasm_err)?;
        Ok(Described {
            meta: world_050::meta(d.meta),
            shape: Some(Shape {
                window: d.window,
                stride: d.stride,
                pure: d.pure,
                one_to_one: d.one_to_one,
            }),
            reads_rows: d.reads_rows,
            forwards_rows: true,
            world: "0.5.0",
            audio_capable: false,
            inputs: 1,
        })
    }

    fn init(
        &self,
        store: &mut Store<Host>,
        format: &Format,
        stream: &StreamInfo,
        name: &str,
        params: &str,
    ) -> Result<Result<(), String>> {
        let video = video_only(format, name)?;
        let wit = world_050::stream_info(stream, format.time_base, name)?;
        self.0
            .ffrwd_av_window_filter()
            .call_init(
                store,
                video.width,
                video.height,
                video.pix_fmt,
                &wit,
                params,
            )
            .map_err(wasm_err)
    }

    fn set_params(&self, store: &mut Store<Host>, params: &str) -> Result<Result<(), String>> {
        self.0
            .ffrwd_av_window_filter()
            .call_set_params(store, params)
            .map_err(wasm_err)
    }

    fn frame_independent(&self, _store: &mut Store<Host>) -> Result<Option<bool>> {
        Ok(None)
    }

    fn process(
        &self,
        store: &mut Store<Host>,
        _format: &Format,
        frames: &[Frame],
        _trailing: &[String],
        last: bool,
        same: SameRule<'_>,
    ) -> Result<Processed> {
        use world_050::window::exports::ffrwd::av::window_filter::InFrame;
        let window: Vec<InFrame> = frames
            .iter()
            .map(|f| InFrame {
                pts: f.pts,
                frame: f.data.clone(),
                rows: f.rows.clone(),
            })
            .collect();
        let out = self
            .0
            .ffrwd_av_window_filter()
            .call_process(store, &window, last)
            .map_err(wasm_err)?;
        Ok(Processed {
            frames: resolve_050(out, frames, same)?,
            trailing: Vec::new(),
        })
    }
}

/// One `same` payload resolved to the pixels behind it. Each world spells the
/// variant separately, so each has its own walk.
macro_rules! resolve_frames {
    ($name:ident, $world:ident) => {
        fn $name(
            produced: Vec<$world::window::exports::ffrwd::av::window_filter::OutFrame>,
            frames: &[Frame],
            same: SameRule<'_>,
        ) -> Result<Vec<Frame>> {
            use $world::window::exports::ffrwd::av::window_filter::FramePayload;
            produced
                .into_iter()
                .map(|frame| {
                    let data = match frame.frame {
                        FramePayload::New(data) => data,
                        FramePayload::Same => unchanged(frames, frame.pts, same)?,
                    };
                    Ok(Frame {
                        pts: frame.pts,
                        data,
                        rows: frame.rows,
                    })
                })
                .collect()
        }
    };
}

resolve_frames!(resolve_0100, world_0100);
resolve_frames!(resolve_090, world_090);
resolve_frames!(resolve_080, world_080);
resolve_frames!(resolve_070, world_070);
resolve_frames!(resolve_060, world_060);
resolve_frames!(resolve_050, world_050);

/// One instantiated filter. Single-threaded by contract: run many `Filter`s in
/// parallel rather than sharing one.
pub struct Filter {
    store: Store<Host>,
    instance: Box<dyn Adapter>,
    meta: Meta,
    shape: Shape,
    reads_rows: bool,
    forwards_rows: bool,
    /// Streams this module reads. Above 1, a `process_window` call is the
    /// pads themselves rather than a window.
    inputs: u32,
    format: Format,
    /// Highest timestamp this instance has produced, for the non-decreasing
    /// check across calls.
    last_pts: Option<i64>,
    /// Where the audio this instance has produced runs to, in ticks: the
    /// timestamp the next sample must carry for the run to be continuous.
    next_pts: Option<i64>,
    /// Samples consumed and samples returned over this instance's life, for
    /// the one-to-one check an audio module is held to.
    samples_in: u64,
    samples_out: u64,
    /// Whether the final call has been made, which may happen once.
    finished: bool,
}

/// A compiled, instantiated and described component, before `init`.
struct Opened {
    store: Store<Host>,
    instance: Box<dyn Adapter>,
    described: Described,
}

/// Compiles, instantiates and calls `describe()` on the component at
/// `module_path`. Shared by `Filter::open` (which goes on to call `init`)
/// and standalone introspection, which needs nothing past `describe()`.
fn instantiate(module_path: &str, purpose: Purpose) -> Result<Opened> {
    let component = compile(module_path)?;
    check_frame_export(&component, module_path)?;

    let (linker, nn) = link(&component, module_path, purpose)?;

    let policy = egress::net_policy()?;
    let wasi = wasi_ctx(granted(module_path)?, policy);
    let mut store = Store::new(
        engine(),
        Host {
            wasi,
            table: ResourceTable::new(),
            nn,
            http: WasiHttpCtx::new(),
            hooks: egress::Hooks::new(policy),
        },
    );

    // Which world and which shape the component fits is read off its exports,
    // not declared. The newest world wins, so a module built against the
    // current one never goes through an adapter.
    let context = || format!("instantiating {module_path}");
    let instance: Box<dyn Adapter> =
        if has_export(&component, &interface("window-filter", "0.10.0")) {
            Box::new(Window0100(
                world_0100::window::WindowModule::instantiate(&mut store, &component, &linker)
                    .map_err(wasm_err)
                    .with_context(context)?,
            ))
        } else if has_export(&component, &interface("window-filter", "0.9.0")) {
            Box::new(Window090(
                world_090::window::WindowModule::instantiate(&mut store, &component, &linker)
                    .map_err(wasm_err)
                    .with_context(context)?,
            ))
        } else if has_export(&component, &interface("window-filter", "0.8.0")) {
            Box::new(Window080(
                world_080::window::WindowModule::instantiate(&mut store, &component, &linker)
                    .map_err(wasm_err)
                    .with_context(context)?,
            ))
        } else if has_export(&component, &interface("window-filter", "0.7.0")) {
            Box::new(Window070(
                world_070::window::WindowModule::instantiate(&mut store, &component, &linker)
                    .map_err(wasm_err)
                    .with_context(context)?,
            ))
        } else if has_export(&component, &interface("window-filter", "0.6.0")) {
            Box::new(Window060(
                world_060::window::WindowModule::instantiate(&mut store, &component, &linker)
                    .map_err(wasm_err)
                    .with_context(context)?,
            ))
        } else if has_export(&component, &interface("window-filter", "0.5.0")) {
            Box::new(Window050(
                world_050::window::WindowModule::instantiate(&mut store, &component, &linker)
                    .map_err(wasm_err)
                    .with_context(context)?,
            ))
        } else if has_export(&component, &interface("meta-filter", "0.10.0")) {
            Box::new(Meta0100(
                world_0100::meta::MetaModule::instantiate(&mut store, &component, &linker)
                    .map_err(wasm_err)
                    .with_context(context)?,
            ))
        } else if has_export(&component, &interface("meta-filter", "0.9.0")) {
            Box::new(Meta090(
                world_090::meta::MetaModule::instantiate(&mut store, &component, &linker)
                    .map_err(wasm_err)
                    .with_context(context)?,
            ))
        } else if has_export(&component, &interface("meta-filter", "0.8.0")) {
            Box::new(Meta080(
                world_080::meta::MetaModule::instantiate(&mut store, &component, &linker)
                    .map_err(wasm_err)
                    .with_context(context)?,
            ))
        } else if has_export(&component, &interface("meta-filter", "0.7.0")) {
            Box::new(Meta070(
                world_070::meta::MetaModule::instantiate(&mut store, &component, &linker)
                    .map_err(wasm_err)
                    .with_context(context)?,
            ))
        } else if has_export(&component, &interface("meta-filter", "0.6.0")) {
            Box::new(Meta060(
                world_060::meta::MetaModule::instantiate(&mut store, &component, &linker)
                    .map_err(wasm_err)
                    .with_context(context)?,
            ))
        } else if has_export(&component, &interface("meta-filter", "0.5.0")) {
            Box::new(Meta050(
                world_050::meta::MetaModule::instantiate(&mut store, &component, &linker)
                    .map_err(wasm_err)
                    .with_context(context)?,
            ))
        } else if has_export(&component, &interface("meta-filter", "0.4.0")) {
            Box::new(Meta040(
                world_040::meta::MetaModule::instantiate(&mut store, &component, &linker)
                    .map_err(wasm_err)
                    .with_context(context)?,
            ))
        } else if has_export(&component, &interface("filter", "0.10.0")) {
            Box::new(Video0100(
                world_0100::video::VideoModule::instantiate(&mut store, &component, &linker)
                    .map_err(wasm_err)
                    .with_context(context)?,
            ))
        } else if has_export(&component, &interface("filter", "0.9.0")) {
            Box::new(Video090(
                world_090::video::VideoModule::instantiate(&mut store, &component, &linker)
                    .map_err(wasm_err)
                    .with_context(context)?,
            ))
        } else if has_export(&component, &interface("filter", "0.8.0")) {
            Box::new(Video080(
                world_080::video::VideoModule::instantiate(&mut store, &component, &linker)
                    .map_err(wasm_err)
                    .with_context(context)?,
            ))
        } else if has_export(&component, &interface("filter", "0.7.0")) {
            Box::new(Video070(
                world_070::video::VideoModule::instantiate(&mut store, &component, &linker)
                    .map_err(wasm_err)
                    .with_context(context)?,
            ))
        } else if has_export(&component, &interface("filter", "0.6.0")) {
            Box::new(Video060(
                world_060::video::VideoModule::instantiate(&mut store, &component, &linker)
                    .map_err(wasm_err)
                    .with_context(context)?,
            ))
        } else if has_export(&component, &interface("filter", "0.5.0")) {
            Box::new(Video050(
                world_050::video::VideoModule::instantiate(&mut store, &component, &linker)
                    .map_err(wasm_err)
                    .with_context(context)?,
            ))
        } else if has_export(&component, &interface("filter", "0.4.0")) {
            Box::new(Video040(
                world_040::video::VideoModule::instantiate(&mut store, &component, &linker)
                    .map_err(wasm_err)
                    .with_context(context)?,
            ))
        } else if has_export(&component, &interface("filter", "0.3.0")) {
            Box::new(Video030(
                world_030::video::VideoModule::instantiate(&mut store, &component, &linker)
                    .map_err(wasm_err)
                    .with_context(context)?,
            ))
        } else {
            Box::new(Video020(
                world_020::video::VideoModule::instantiate(&mut store, &component, &linker)
                    .map_err(wasm_err)
                    .with_context(context)?,
            ))
        };

    let described = instance.described(&mut store)?;
    if let Some(shape) = described.shape {
        check_shape(&shape, described.inputs, &described.meta.name)?;
    }

    Ok(Opened {
        store,
        instance,
        described,
    })
}

/// A window of zero frames is no window, and a stride past the window would
/// step over frames the module never saw. A module reading several streams
/// takes one frame off each pad and hands one back, so it is window 1,
/// stride 1 and nothing else.
fn check_shape(shape: &Shape, inputs: u32, name: &str) -> Result<()> {
    if shape.window == 0 {
        bail!("{name} publishes window 0, which is nothing at all");
    }
    if shape.stride == 0 {
        bail!("{name} publishes stride 0, which never advances");
    }
    if shape.stride > shape.window {
        bail!(
            "{name} publishes stride {} past its window {}, which would step over what it never saw",
            shape.stride,
            shape.window
        );
    }
    if inputs == 0 {
        bail!("{name} publishes inputs 0, and a module reads at least one stream");
    }
    if inputs > 1 && (shape.window != 1 || shape.stride != 1) {
        bail!(
            "{name} reads {inputs} streams over a window of {} every {}; a module reading several \
             streams takes one frame off each and hands one back, so its window and stride are 1",
            shape.window,
            shape.stride
        );
    }
    Ok(())
}

/// Compiles and instantiates the component at `module_path` far enough to
/// call `describe()`, without `init`-ing an instance for any particular
/// stream or frame geometry. For introspection ahead of a real run.
pub fn describe(module_path: &str) -> Result<Described> {
    Ok(instantiate(module_path, Purpose::Describe)?.described)
}

/// One instantiated values module, in whichever world it was built against.
enum ValuesInstance {
    W0100(world_0100::values::ValuesModule),
    W090(world_090::values::ValuesModule),
    W080(world_080::values::ValuesModule),
    W070(world_070::values::ValuesModule),
    W060(world_060::values::ValuesModule),
    W050(world_050::values::ValuesModule),
    W040(world_040::values::ValuesModule),
    W030(world_030::values::ValuesModule),
}

impl ValuesInstance {
    fn list_functions(&self, store: &mut Store<Host>) -> Result<Vec<FunctionMeta>> {
        // Each world spells `function-meta` separately, so each arm carries
        // its own into the host's.
        macro_rules! listed {
            ($b:expr) => {
                $b.ffrwd_av_values()
                    .call_list_functions(store)
                    .map_err(wasm_err)?
                    .into_iter()
                    .map(|f| FunctionMeta {
                        name: f.name,
                        params_schema: f.params_schema,
                        result_schema: f.result_schema,
                    })
                    .collect()
            };
        }
        Ok(match self {
            ValuesInstance::W0100(b) => listed!(b),
            ValuesInstance::W090(b) => listed!(b),
            ValuesInstance::W080(b) => listed!(b),
            ValuesInstance::W070(b) => listed!(b),
            ValuesInstance::W060(b) => listed!(b),
            ValuesInstance::W050(b) => listed!(b),
            ValuesInstance::W040(b) => listed!(b),
            ValuesInstance::W030(b) => listed!(b),
        })
    }

    fn invoke(
        &self,
        store: &mut Store<Host>,
        name: &str,
        args: &str,
    ) -> Result<Result<String, String>> {
        match self {
            ValuesInstance::W0100(b) => b.ffrwd_av_values().call_invoke(store, name, args),
            ValuesInstance::W090(b) => b.ffrwd_av_values().call_invoke(store, name, args),
            ValuesInstance::W080(b) => b.ffrwd_av_values().call_invoke(store, name, args),
            ValuesInstance::W070(b) => b.ffrwd_av_values().call_invoke(store, name, args),
            ValuesInstance::W060(b) => b.ffrwd_av_values().call_invoke(store, name, args),
            ValuesInstance::W050(b) => b.ffrwd_av_values().call_invoke(store, name, args),
            ValuesInstance::W040(b) => b.ffrwd_av_values().call_invoke(store, name, args),
            ValuesInstance::W030(b) => b.ffrwd_av_values().call_invoke(store, name, args),
        }
        .map_err(wasm_err)
    }
}

/// Compiles and instantiates the component at `module_path` against the
/// values world it was built for. Shared by `list_functions` and `invoke`.
fn instantiate_values(
    module_path: &str,
    purpose: Purpose,
) -> Result<(Store<Host>, ValuesInstance)> {
    let component = compile(module_path)?;
    check_values_export(&component, module_path)?;

    let (linker, nn) = link(&component, module_path, purpose)?;

    let policy = egress::net_policy()?;
    let wasi = wasi_ctx(granted(module_path)?, policy);
    let mut store = Store::new(
        engine(),
        Host {
            wasi,
            table: ResourceTable::new(),
            nn,
            http: WasiHttpCtx::new(),
            hooks: egress::Hooks::new(policy),
        },
    );

    let context = || format!("instantiating {module_path}");
    let instance = if has_export(&component, &interface("values", "0.10.0")) {
        ValuesInstance::W0100(
            world_0100::values::ValuesModule::instantiate(&mut store, &component, &linker)
                .map_err(wasm_err)
                .with_context(context)?,
        )
    } else if has_export(&component, &interface("values", "0.9.0")) {
        ValuesInstance::W090(
            world_090::values::ValuesModule::instantiate(&mut store, &component, &linker)
                .map_err(wasm_err)
                .with_context(context)?,
        )
    } else if has_export(&component, &interface("values", "0.8.0")) {
        ValuesInstance::W080(
            world_080::values::ValuesModule::instantiate(&mut store, &component, &linker)
                .map_err(wasm_err)
                .with_context(context)?,
        )
    } else if has_export(&component, &interface("values", "0.7.0")) {
        ValuesInstance::W070(
            world_070::values::ValuesModule::instantiate(&mut store, &component, &linker)
                .map_err(wasm_err)
                .with_context(context)?,
        )
    } else if has_export(&component, &interface("values", "0.6.0")) {
        ValuesInstance::W060(
            world_060::values::ValuesModule::instantiate(&mut store, &component, &linker)
                .map_err(wasm_err)
                .with_context(context)?,
        )
    } else if has_export(&component, &interface("values", "0.5.0")) {
        ValuesInstance::W050(
            world_050::values::ValuesModule::instantiate(&mut store, &component, &linker)
                .map_err(wasm_err)
                .with_context(context)?,
        )
    } else if has_export(&component, &interface("values", "0.4.0")) {
        ValuesInstance::W040(
            world_040::values::ValuesModule::instantiate(&mut store, &component, &linker)
                .map_err(wasm_err)
                .with_context(context)?,
        )
    } else {
        ValuesInstance::W030(
            world_030::values::ValuesModule::instantiate(&mut store, &component, &linker)
                .map_err(wasm_err)
                .with_context(context)?,
        )
    };

    Ok((store, instance))
}

/// The value functions a module publishes, from its `list-functions()`. The
/// module must export the values interface; a compile-time property callers
/// check with `exports_values` before calling into a module by name.
pub fn list_functions(module_path: &str) -> Result<Vec<FunctionMeta>> {
    let (mut store, instance) = instantiate_values(module_path, Purpose::Describe)?;
    instance.list_functions(&mut store)
}

/// Calls one value function at compile time. `args` is one JSON object keyed
/// by parameter name. The outer `Result` is a host-side failure (compiling,
/// instantiating, a wasm trap); the inner one is the module's own outcome.
pub fn invoke(module_path: &str, name: &str, args: &str) -> Result<Result<String, String>> {
    let (mut store, instance) = instantiate_values(module_path, Purpose::Run)?;
    instance.invoke(&mut store, name, args)
}

/// The packet-sink export's full name. Only the current world carries the
/// interface, so a packet sink is never adapted.
fn packet_interface() -> String {
    interface("packet-sink", WORLD)
}

/// Whether the component at `module_path` exports the packet-sink interface,
/// and so consumes encoded packets rather than decoded frames.
pub fn exports_packet_sink(module_path: &str) -> Result<bool> {
    let component = compile(module_path)?;
    Ok(has_export(&component, &packet_interface()))
}

/// Errors naming the component's actual exports when the packet-sink
/// interface is missing.
fn check_packet_export(component: &Component, module_path: &str) -> Result<()> {
    if has_export(component, &packet_interface()) {
        return Ok(());
    }
    let wanted = packet_interface();
    let exports = component_exports(component);
    if exports.is_empty() {
        bail!("{module_path} exports nothing, so not {wanted}");
    }
    bail!(
        "{module_path} does not export {wanted}; it exports {}",
        exports.join(", ")
    );
}

/// Compiles and instantiates the component at `module_path` against the
/// packet world. Shared by `describe_packet_sink` and `PacketSink::open`.
fn instantiate_packet(
    module_path: &str,
    purpose: Purpose,
) -> Result<(Store<Host>, world_0100::packet::PacketModule)> {
    let component = compile(module_path)?;
    check_packet_export(&component, module_path)?;

    let (linker, nn) = link(&component, module_path, purpose)?;

    let policy = egress::net_policy()?;
    let wasi = wasi_ctx(granted(module_path)?, policy);
    let mut store = Store::new(
        engine(),
        Host {
            wasi,
            table: ResourceTable::new(),
            nn,
            http: WasiHttpCtx::new(),
            hooks: egress::Hooks::new(policy),
        },
    );
    let module = world_0100::packet::PacketModule::instantiate(&mut store, &component, &linker)
        .map_err(wasm_err)
        .with_context(|| format!("instantiating {module_path}"))?;
    Ok((store, module))
}

/// Compiles and instantiates the component at `module_path` far enough to
/// call the packet sink's `describe()`, without opening it for any stream.
pub fn describe_packet_sink(module_path: &str) -> Result<DescribedPacketSink> {
    let (mut store, module) = instantiate_packet(module_path, Purpose::Describe)?;
    let d = module
        .ffrwd_av_packet_sink()
        .call_describe(&mut store)
        .map_err(wasm_err)?;
    Ok(DescribedPacketSink {
        meta: world_0100::meta(d.meta),
        codecs: d.codecs,
        world: WORLD,
    })
}

/// One instantiated packet sink: encoded packets in, rows out, nothing else
/// leaving. Single-threaded by contract, like [`Filter`].
pub struct PacketSink {
    store: Store<Host>,
    module: world_0100::packet::PacketModule,
    meta: Meta,
    /// Whether the final call has been made, which may happen once.
    finished: bool,
}

impl PacketSink {
    /// Compiles (cached process-wide by path) and instantiates the component
    /// at `module_path`, then calls the sink's `init`. The stream's codec
    /// must be one the sink accepts.
    pub fn open(
        module_path: &str,
        stream: &CodedStream,
        info: &StreamInfo,
        params: &str,
    ) -> Result<PacketSink> {
        let (mut store, module) = instantiate_packet(module_path, Purpose::Run)?;
        let d = module
            .ffrwd_av_packet_sink()
            .call_describe(&mut store)
            .map_err(wasm_err)?;
        let meta = world_0100::meta(d.meta);
        if !d.codecs.is_empty() && !d.codecs.iter().any(|c| c == &stream.codec) {
            bail!(
                "{} does not accept {}; it publishes {}",
                meta.name,
                stream.codec,
                d.codecs.join(", ")
            );
        }

        let (num, den) = stream.time_base.rational(&meta.name)?;
        let wit_stream = world_0100::packet::exports::ffrwd::av::packet_sink::CodedStream {
            codec: stream.codec.clone(),
            time_base: world_0100::video::ffrwd::av::types::Rational { num, den },
            width: stream.width,
            height: stream.height,
            extradata: stream.extradata.clone(),
        };
        let wit_info = world_0100::stream_info(info, stream.time_base, &meta.name)?;
        module
            .ffrwd_av_packet_sink()
            .call_init(&mut store, &wit_stream, &wit_info, params)
            .map_err(wasm_err)?
            .map_err(|e| anyhow!("{} rejected params: {e}", meta.name))?;

        Ok(PacketSink {
            store,
            module,
            meta,
            finished: false,
        })
    }

    /// The sink's `describe()`, read once at open.
    pub fn meta(&self) -> &Meta {
        &self.meta
    }

    /// Module name from `describe()`, for error messages.
    pub fn name(&self) -> &str {
        &self.meta.name
    }

    /// Replaces the sink's parameters between calls.
    pub fn set_params(&mut self, params: &str) -> Result<()> {
        self.module
            .ffrwd_av_packet_sink()
            .call_set_params(&mut self.store, params)
            .map_err(wasm_err)?
            .map_err(|e| anyhow!("{} rejected params: {e}", self.meta.name))?;
        Ok(())
    }

    /// Packets through the sink, in decode order. `last` marks the final
    /// call, which may carry no packets at all and happens once. Only the
    /// final call may return trailing rows.
    pub fn process(&mut self, packets: &[Packet], last: bool) -> Result<Emitted> {
        if self.finished {
            bail!(
                "{}: called again after the final call, which happens once",
                self.meta.name
            );
        }
        self.finished = last;

        let wit: Vec<world_0100::packet::exports::ffrwd::av::packet_sink::Packet> = packets
            .iter()
            .map(
                |p| world_0100::packet::exports::ffrwd::av::packet_sink::Packet {
                    pts: p.pts,
                    dts: p.dts,
                    keyframe: p.keyframe,
                    data: p.data.clone(),
                },
            )
            .collect();
        let produced = self
            .module
            .ffrwd_av_packet_sink()
            .call_process(&mut self.store, &wit, last)
            .map_err(wasm_err)?;
        if !last && !produced.trailing.is_empty() {
            bail!(
                "{} returned {} trailing row(s) from a call that is not its final one; only the \
                 final call may",
                self.meta.name,
                produced.trailing.len()
            );
        }
        Ok(Emitted {
            rows: produced.rows,
            trailing: produced.trailing,
        })
    }
}

impl Filter {
    /// Compiles (cached process-wide by path) and instantiates the component at
    /// `module_path`, then calls the module's `init`. The stream must be the
    /// kind the module publishes, in a format it accepts.
    pub fn open(
        module_path: &str,
        format: &Format,
        stream: &StreamInfo,
        params: &str,
    ) -> Result<Filter> {
        let Opened {
            mut store,
            instance,
            described,
        } = instantiate(module_path, Purpose::Run)?;

        check_audio_capable(&described, format)?;
        let declared = described.meta.kind()?;
        check_accepts(&described.meta, declared, format)?;

        let Described {
            meta,
            shape,
            reads_rows,
            forwards_rows,
            inputs,
            ..
        } = described;

        instance
            .init(&mut store, format, stream, &meta.name, params)?
            .map_err(|e| anyhow!("{} rejected params: {e}", meta.name))?;

        // A per-frame module answers purity only once it is init-ed, and the
        // answer may depend on the parameters it was given.
        let shape = match shape {
            Some(shape) => shape,
            None => Shape {
                pure: instance
                    .frame_independent(&mut store)?
                    .expect("a per-frame module answers purity at runtime"),
                ..ADAPTED_SHAPE
            },
        };

        Ok(Filter {
            store,
            instance,
            meta,
            shape,
            reads_rows,
            forwards_rows,
            inputs,
            format: *format,
            last_pts: None,
            next_pts: None,
            samples_in: 0,
            samples_out: 0,
            finished: false,
        })
    }

    /// Whether this module acts on the rows an upstream module emitted. A
    /// windowed module declares it; a per-frame one answers it by exporting
    /// `meta-filter` or not.
    pub fn reads_rows(&self) -> bool {
        self.reads_rows
    }

    /// Whether upstream rows may leave on this module's own output frames.
    pub fn forwards_rows(&self) -> bool {
        self.forwards_rows
    }

    /// The module's `describe()`, read once at open.
    pub fn meta(&self) -> &Meta {
        &self.meta
    }

    /// Module name from `describe()`, for error messages.
    pub fn name(&self) -> &str {
        &self.meta.name
    }

    /// How the host must drive this module.
    pub fn shape(&self) -> Shape {
        self.shape
    }

    /// How many streams this module reads. Above 1, one `process_window` call
    /// carries one frame per pad rather than a window of one stream.
    pub fn inputs(&self) -> u32 {
        self.inputs
    }

    /// Replaces the module's parameters. The instance keeps its state and the
    /// frame size is unchanged.
    pub fn set_params(&mut self, params: &str) -> Result<()> {
        self.instance
            .set_params(&mut self.store, params)?
            .map_err(|e| anyhow!("{} rejected params: {e}", self.meta.name))?;
        Ok(())
    }

    /// One window through the module. `last` marks the final call, which
    /// carries whatever the last stride left over and happens once.
    /// `trailing` is the rows an upstream module had no frame to put them on,
    /// which only the final call may carry.
    ///
    /// Frames the module returns unchanged are resolved to their pixels here,
    /// so a caller never has to look back at the window.
    pub fn process_window(
        &mut self,
        frames: &[Frame],
        trailing: &[String],
        last: bool,
    ) -> Result<Processed> {
        if self.finished {
            bail!(
                "{}: called again after the final window, which happens once",
                self.meta.name
            );
        }
        if !last && !trailing.is_empty() {
            bail!(
                "{}: handed {} trailing row(s) before its final window; only the final call carries them",
                self.meta.name,
                trailing.len()
            );
        }
        let arriving = self.check_window(frames)?;
        self.finished = last;

        // Borrowed field by field: the same-rule names the module, and the
        // call needs the store at the same time.
        let Filter {
            store,
            instance,
            meta,
            shape,
            format,
            ..
        } = self;
        let same = if format.audio().is_some() && shape.stride != shape.window {
            SameRule::RefusedForOverlap(&meta.name)
        } else {
            SameRule::Allowed
        };
        let out = instance.process(store, format, frames, trailing, last, same)?;

        if !last && !out.trailing.is_empty() {
            bail!(
                "{} returned {} trailing row(s) from a window that is not its final one; only the final call may",
                self.meta.name,
                out.trailing.len()
            );
        }
        for frame in &out.frames {
            if let Some(video) = self.format.video() {
                if frame.data.len() != video.frame_len {
                    bail!(
                        "{} returned {} bytes at pts {}, expected {}",
                        self.meta.name,
                        frame.data.len(),
                        frame.pts,
                        video.frame_len
                    );
                }
            }
            if let Some(previous) = self.last_pts {
                if frame.pts < previous {
                    bail!(
                        "{} returned pts {} after pts {}; output timestamps never decrease",
                        self.meta.name,
                        frame.pts,
                        previous
                    );
                }
            }
            self.last_pts = Some(frame.pts);
        }

        if let Some(audio) = self.format.audio() {
            // Consumed is the stride, not the window: overlapping windows hand
            // the same samples over more than once, and only one call's worth
            // of them leaves the instance.
            self.samples_in += if last {
                arriving
            } else {
                u64::from(self.shape.stride).min(arriving)
            };
            self.check_audio_output(audio, &out.frames, last)?;
        }
        Ok(out)
    }

    /// What one window may hold, and how many samples it held. Every refusal
    /// names the module.
    fn check_window(&self, frames: &[Frame]) -> Result<u64> {
        if self.inputs > 1 {
            return self.check_pads(frames);
        }
        match self.format.media {
            Media::Video(video) => {
                if frames.len() > self.shape.window as usize {
                    bail!(
                        "{}: handed {} frames for a window of {}",
                        self.meta.name,
                        frames.len(),
                        self.shape.window
                    );
                }
                for frame in frames {
                    if frame.data.len() != video.frame_len {
                        bail!(
                            "{}: expected {} byte frames, got {} at pts {}",
                            self.meta.name,
                            video.frame_len,
                            frame.data.len(),
                            frame.pts
                        );
                    }
                }
                Ok(frames.len() as u64)
            }
            Media::Audio(audio) => {
                if frames.len() > 1 {
                    bail!(
                        "{}: handed {} pieces of audio for one window; a window arrives whole",
                        self.meta.name,
                        frames.len()
                    );
                }
                let Some(frame) = frames.first() else {
                    return Ok(0);
                };
                let samples = self.samples(audio, frame.data.len(), frame.pts)?;
                if samples > u64::from(self.shape.window) {
                    bail!(
                        "{}: handed {samples} samples for a window of {}",
                        self.meta.name,
                        self.shape.window
                    );
                }
                Ok(samples)
            }
        }
    }

    /// What one call to a module reading several streams may hold: one frame
    /// per pad, in pad order, every one at the same timestamp. The final call
    /// carries nothing, since window and stride are 1 and nothing is left
    /// over. Every refusal names the module and the pad.
    fn check_pads(&self, frames: &[Frame]) -> Result<u64> {
        if frames.is_empty() {
            return Ok(0);
        }
        if frames.len() != self.inputs as usize {
            bail!(
                "{}: handed {} frame(s) for the {} stream(s) it reads; a call carries one frame \
                 per pad",
                self.meta.name,
                frames.len(),
                self.inputs
            );
        }
        let head = frames[0].pts;
        for (pad, frame) in frames.iter().enumerate().skip(1) {
            if frame.pts != head {
                bail!(
                    "{}: pad 0 is at pts {head} and pad {pad} at pts {}; the pads of one call \
                     carry the same timestamp",
                    self.meta.name,
                    frame.pts
                );
            }
        }
        if let Some(video) = self.format.video() {
            for (pad, frame) in frames.iter().enumerate() {
                if frame.data.len() != video.frame_len {
                    bail!(
                        "{}: expected {} byte frames, got {} on pad {pad} at pts {}",
                        self.meta.name,
                        video.frame_len,
                        frame.data.len(),
                        frame.pts
                    );
                }
            }
            return Ok(1);
        }
        let audio = self
            .format
            .audio()
            .expect("an instance is opened for video or for audio");
        self.samples(audio, frames[0].data.len(), head)
    }

    /// Samples in a payload of `bytes`, refusing a payload that is not a whole
    /// number of them.
    fn samples(&self, audio: AudioFormat, bytes: usize, pts: i64) -> Result<u64> {
        let width = audio.sample_len();
        if !bytes.is_multiple_of(width) {
            bail!(
                "{}: {bytes} bytes at pts {pts} is not a whole number of samples; one sample of \
                 {} across {} channel(s) is {width} bytes",
                self.meta.name,
                audio.sample_fmt,
                audio.channels
            );
        }
        Ok((bytes / width) as u64)
    }

    /// The samples leaving must be whole, and - for a one-to-one module - must
    /// run on from where the last call left off and, by the final call, add up
    /// to the samples that arrived.
    fn check_audio_output(
        &mut self,
        audio: AudioFormat,
        frames: &[Frame],
        last: bool,
    ) -> Result<()> {
        for frame in frames {
            let samples = self.samples(audio, frame.data.len(), frame.pts)?;
            self.samples_out += samples;
            if !self.shape.one_to_one {
                continue;
            }
            let expected = self.next_pts.unwrap_or(frame.pts);
            if frame.pts != expected {
                bail!(
                    "{} returned samples at pts {} where its output so far ends at pts {expected}; \
                     a one-to-one audio module leaves no gap and no overlap",
                    self.meta.name,
                    frame.pts
                );
            }
            self.next_pts = Some(expected + self.ticks(audio, samples)?);
        }
        if last && self.shape.one_to_one && self.samples_in != self.samples_out {
            bail!(
                "{} was handed {} samples and returned {}; a one-to-one audio module returns the \
                 samples it was handed",
                self.meta.name,
                self.samples_in,
                self.samples_out
            );
        }
        Ok(())
    }

    /// The span `samples` cover, in ticks of the stream's time base. At the
    /// natural base of 1/sample-rate one tick is one sample.
    fn ticks(&self, audio: AudioFormat, samples: u64) -> Result<i64> {
        let base = self.format.time_base;
        let num = i128::from(samples) * i128::from(base.den);
        let den = i128::from(audio.sample_rate) * i128::from(base.num);
        if den == 0 || num % den != 0 {
            bail!(
                "{}: {samples} samples at {} Hz do not land on a whole tick of the time base {}/{}",
                self.meta.name,
                audio.sample_rate,
                base.num,
                base.den
            );
        }
        i64::try_from(num / den)
            .map_err(|_| anyhow!("{}: {samples} samples overflow a timestamp", self.meta.name))
    }
}

/// An audio stream reaching an interface that cannot carry one. Every world
/// before 0.7.0 hosts video alone, and so does the per-frame interface in
/// every world, the current one included.
fn check_audio_capable(described: &Described, format: &Format) -> Result<()> {
    if format.audio().is_none() || described.audio_capable {
        return Ok(());
    }
    let name = &described.meta.name;
    if described.shape.is_none() {
        bail!(
            "{name} exports the per-frame filter interface, which hosts video alone, and this \
             stream is audio"
        );
    }
    bail!(
        "{name} is a module of ffrwd:av@{}, which hosts video alone, and this stream is audio",
        described.world
    );
}

/// The stream must be the kind the module publishes, in a format it accepts.
/// Every refusal names the module and what it does publish.
fn check_accepts(meta: &Meta, declared: Kind, format: &Format) -> Result<()> {
    match format.media {
        Media::Video(video) => {
            if declared != Kind::Video {
                bail!(
                    "{} is an audio module and this stream is video; it publishes sample formats {}",
                    meta.name,
                    meta.sample_formats.join(", ")
                );
            }
            if !meta.pixel_formats.iter().any(|f| f == video.pix_fmt) {
                bail!(
                    "{} does not accept pixel format {}; it publishes {}",
                    meta.name,
                    video.pix_fmt,
                    meta.pixel_formats.join(", ")
                );
            }
        }
        Media::Audio(audio) => {
            if declared != Kind::Audio {
                bail!(
                    "{} is a video module and this stream is audio; it publishes pixel formats {}",
                    meta.name,
                    meta.pixel_formats.join(", ")
                );
            }
            if !meta.sample_formats.iter().any(|f| f == audio.sample_fmt) {
                bail!(
                    "{} does not accept sample format {}; it publishes {}",
                    meta.name,
                    audio.sample_fmt,
                    meta.sample_formats.join(", ")
                );
            }
            if !meta.sample_rates.is_empty() && !meta.sample_rates.contains(&audio.sample_rate) {
                bail!(
                    "{} does not accept {} Hz; it publishes {}",
                    meta.name,
                    audio.sample_rate,
                    numbers(&meta.sample_rates)
                );
            }
            if !meta.channel_counts.is_empty() && !meta.channel_counts.contains(&audio.channels) {
                bail!(
                    "{} does not accept {} channel(s); it publishes {}",
                    meta.name,
                    audio.channels,
                    numbers(&meta.channel_counts)
                );
            }
        }
    }
    Ok(())
}

/// A list of numbers as a module publishes them, for a refusal.
fn numbers(values: &[u32]) -> String {
    values
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

/// The bytes behind a `same` payload: the input this call received at the
/// same timestamp, or - when the call received exactly one - that one,
/// whatever timestamp the output carries.
fn unchanged(frames: &[Frame], pts: i64, same: SameRule<'_>) -> Result<Vec<u8>> {
    if let SameRule::RefusedForOverlap(name) = same {
        bail!(
            "{name} passed its window through unchanged, and its windows overlap; every sample \
             would leave more than once, so an audio module may only do that when its stride is \
             its window"
        );
    }
    if let Some(frame) = frames.iter().find(|f| f.pts == pts) {
        return Ok(frame.data.clone());
    }
    if let [only] = frames {
        return Ok(only.data.clone());
    }
    bail!(
        "an unchanged frame at pts {pts} names no input: the call carried {} frames and none of \
         them is at that timestamp",
        frames.len()
    )
}

#[cfg(test)]
mod tests {
    use super::{check_shape, AudioFormat, Kind, Meta, Shape, StreamInfo, TimeBase};

    /// A shape a module might publish.
    fn shape(window: u32, stride: u32) -> Shape {
        Shape {
            window,
            stride,
            pure: true,
            one_to_one: true,
        }
    }

    #[test]
    fn a_module_reading_several_streams_is_one_frame_off_each_and_nothing_else() {
        // A pad has no answer for which of its frames pairs with which of
        // another pad's, so the only shape a multi-pad module may publish is
        // one frame in per pad and one out.
        check_shape(&shape(1, 1), 2, "blur_mask").expect("window 1, stride 1");

        let wide = check_shape(&shape(4, 4), 2, "blur_mask").expect_err("refused");
        let message = wide.to_string();
        assert!(message.contains("blur_mask"), "got: {message}");
        assert!(message.contains("reads 2 streams"), "got: {message}");
        assert!(
            message.contains("window of 4 every 4"),
            "the shape it published is named: {message}"
        );

        let strided = check_shape(&shape(2, 1), 3, "blur_mask").expect_err("refused");
        assert!(
            strided.to_string().contains("window and stride are 1"),
            "got: {strided}"
        );
    }

    #[test]
    fn a_module_reading_one_stream_keeps_every_window_it_could_before() {
        for (window, stride) in [(1, 1), (4, 4), (8, 2)] {
            check_shape(&shape(window, stride), 1, "shots")
                .unwrap_or_else(|e| panic!("window {window} stride {stride} is allowed: {e}"));
        }
    }

    #[test]
    fn a_module_reading_no_stream_at_all_is_refused_by_name() {
        let err = check_shape(&shape(1, 1), 0, "nothing").expect_err("refused");
        let message = err.to_string();
        assert!(message.contains("nothing"), "got: {message}");
        assert!(message.contains("at least one stream"), "got: {message}");
    }

    /// A module description naming the formats of the kinds given.
    fn meta(pixel: &[&str], sample: &[&str]) -> Meta {
        Meta {
            name: "m".to_string(),
            version: "0.1.0".to_string(),
            params_schema: String::new(),
            rows_schema: String::new(),
            pixel_formats: pixel.iter().map(|f| f.to_string()).collect(),
            sample_formats: sample.iter().map(|f| f.to_string()).collect(),
            sample_rates: Vec::new(),
            channel_counts: Vec::new(),
            rows_language: Vec::new(),
        }
    }

    #[test]
    fn which_formats_a_module_names_is_what_says_its_kind() {
        assert_eq!(meta(&["rgba"], &[]).kind().expect("video"), Kind::Video);
        assert_eq!(meta(&[], &["f32"]).kind().expect("audio"), Kind::Audio);
    }

    #[test]
    fn a_module_of_both_kinds_or_neither_is_refused_by_name() {
        let both = meta(&["rgba"], &["f32"]).kind().expect_err("refused");
        assert!(both.to_string().contains("m"), "got: {both}");
        assert!(both.to_string().contains("one kind"), "got: {both}");

        let neither = meta(&[], &[]).kind().expect_err("refused");
        assert!(
            neither.to_string().contains("neither a video module"),
            "got: {neither}"
        );
    }

    #[test]
    fn a_sample_is_every_channel_at_one_instant() {
        let stereo_f32 = AudioFormat {
            sample_rate: 48_000,
            channels: 2,
            sample_fmt: "f32",
        };
        assert_eq!(stereo_f32.sample_len(), 8);
        assert_eq!(
            AudioFormat {
                sample_fmt: "s16",
                ..stereo_f32
            }
            .sample_len(),
            4
        );
        assert_eq!(
            AudioFormat {
                channels: 1,
                ..stereo_f32
            }
            .sample_len(),
            4
        );
    }

    fn a_stream() -> StreamInfo {
        StreamInfo {
            index: 0,
            kind: "video".to_string(),
            codec: "rawvideo".to_string(),
            duration: None,
            tags: vec![("language".to_string(), "eng".to_string())],
        }
    }

    #[test]
    fn the_current_world_is_told_the_time_base_as_a_field_and_nothing_else() {
        let wit = super::world_080::stream_info(&a_stream(), TimeBase { num: 1, den: 25 }, "m")
            .expect("the time base fits");
        assert_eq!(
            wit.tags,
            vec![("language".to_string(), "eng".to_string())],
            "the field replaces the tag, so nothing is stamped on"
        );
        assert_eq!((wit.time_base.num, wit.time_base.den), (1, 25));
    }

    #[test]
    fn a_world_without_the_field_is_told_the_time_base_as_a_tag() {
        let wit = super::world_050::stream_info(&a_stream(), TimeBase { num: 1, den: 25 }, "m")
            .expect("a tag always fits");
        assert_eq!(
            wit.tags,
            vec![
                ("language".to_string(), "eng".to_string()),
                ("time-base".to_string(), "1/25".to_string()),
            ]
        );
    }

    #[test]
    fn the_time_base_field_is_the_numerator_over_the_denominator() {
        assert_eq!(
            TimeBase { num: 1, den: 65536 }.rational("m").expect("fits"),
            (1, 65536)
        );
        assert_eq!(
            TimeBase {
                num: 1001,
                den: 30000
            }
            .rational("m")
            .expect("fits"),
            (1001, 30000)
        );
    }

    #[test]
    fn the_field_and_the_seconds_read_the_ratio_the_same_way_up() {
        // A module converting timestamps from the field must land where the
        // host's own conversion lands, or its seconds are a whole time base
        // out.
        let base = TimeBase { num: 1, den: 25 };
        let (num, den) = base.rational("m").expect("fits");
        assert_eq!((num, den), (1, 25));
        assert!(
            (base.seconds(25) - 1.0).abs() < 1e-12,
            "25 ticks of 1/25 is one second"
        );
        assert!(
            (25.0 * f64::from(num) / f64::from(den) - 1.0).abs() < 1e-12,
            "the field reads the same way up as the host's own conversion"
        );
    }

    #[test]
    fn a_time_base_too_wide_for_the_field_is_refused_by_name() {
        let err = TimeBase {
            num: 1,
            den: u64::from(u32::MAX),
        }
        .rational("wideness")
        .expect_err("refused");
        assert!(err.to_string().contains("wideness"), "got: {err}");
    }

    #[test]
    fn the_time_base_tag_is_the_numerator_over_the_denominator() {
        // The tag is how a module of a world before the field reads the same
        // ratio, through its adapter.
        assert_eq!(TimeBase { num: 1, den: 65536 }.tag(), "1/65536");
        assert_eq!(
            TimeBase {
                num: 1001,
                den: 30000
            }
            .tag(),
            "1001/30000"
        );
    }

    #[test]
    fn the_tag_and_the_seconds_read_the_ratio_the_same_way_up() {
        let base = TimeBase { num: 1, den: 25 };
        assert_eq!(base.tag(), "1/25");
        assert!(
            (base.seconds(25) - 1.0).abs() < 1e-12,
            "25 ticks of 1/25 is one second"
        );
    }
}

// Proves the network grant end to end through this module's own `link()` +
// `wasi_ctx()` path, and that it is per module. The subject is a wasip2
// command component (built by the spike workspace under spikes/quinn-wasi)
// that creates a UDP socket, sends a datagram to 127.0.0.1:39091 and expects
// it echoed back. It exports `wasi:cli/run`, not `ffrwd:av`, so it is driven
// generically here rather than through an adapter. The test skips when the
// artifact has not been built.
#[cfg(test)]
mod net_grant_test {
    use super::*;
    use wasmtime::component::ComponentExportIndex;

    fn udp_component() -> Option<PathBuf> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../spikes/quinn-wasi/q3-udp/target/wasm32-wasip2/debug/q3-udp.wasm");
        path.exists().then_some(path)
    }

    /// Instantiates the component with the grants its path currently holds
    /// and calls its `wasi:cli/run` export. The policy is a parameter so the
    /// test never touches the process environment.
    fn run_component(path: &Path, policy: NetPolicy) -> Result<()> {
        let module_path = path.display().to_string();
        let component = Component::from_file(engine(), path).map_err(wasm_err)?;
        let (linker, nn) = link(&component, &module_path, Purpose::Describe)?;
        let mut store = Store::new(
            engine(),
            Host {
                wasi: wasi_ctx(granted(&module_path)?, policy),
                table: ResourceTable::new(),
                nn,
                http: WasiHttpCtx::new(),
                hooks: egress::Hooks::new(policy),
            },
        );
        let instance = linker
            .instantiate(&mut store, &component)
            .map_err(wasm_err)?;

        let run_interface = component_exports(&component)
            .into_iter()
            .find(|name| name.starts_with("wasi:cli/run@"))
            .ok_or_else(|| anyhow!("component exports no wasi:cli/run"))?;
        let interface: ComponentExportIndex = instance
            .get_export_index(&mut store, None, &run_interface)
            .ok_or_else(|| anyhow!("{run_interface} not found on instance"))?;
        let func = instance
            .get_export_index(&mut store, Some(&interface), "run")
            .ok_or_else(|| anyhow!("run not found in {run_interface}"))?;
        let run = instance
            .get_typed_func::<(), (Result<(), ()>,)>(&mut store, func)
            .map_err(wasm_err)?;
        let (result,) = run.call(&mut store, ()).map_err(wasm_err)?;
        result.map_err(|()| anyhow!("run returned failure"))
    }

    #[test]
    fn the_network_is_refused_without_the_grant_and_reachable_with_it() {
        let Some(path) = udp_component() else {
            eprintln!("udp component not built; skipping");
            return;
        };

        let echo = std::net::UdpSocket::bind("127.0.0.1:39091").expect("bind echo peer");
        echo.set_read_timeout(Some(std::time::Duration::from_secs(10)))
            .expect("set echo timeout");
        let peer = std::thread::spawn(move || {
            let mut buf = [0u8; 2048];
            let (n, from) = echo.recv_from(&mut buf).ok()?;
            echo.send_to(&buf[..n], from).ok()?;
            Some(n)
        });

        // Without any grant: sockets are linked, but creation is refused with
        // access-denied, on which the module aborts. Order matters — grants
        // accumulate for the run, so deny must be proven first.
        let denied = run_component(&path, NetPolicy::Unrestricted);
        assert!(
            denied.is_err(),
            "module reached wasi:sockets without a grant"
        );

        // A grant naming a DIFFERENT module leaves this one refused.
        grant_net("some/other/module.wasm").expect("record the other grant");
        let still_denied = run_component(&path, NetPolicy::Unrestricted);
        assert!(
            still_denied.is_err(),
            "a grant is per module, and this one names a different module"
        );

        // With its own grant: the same module round-trips its datagram.
        grant_net(&path.display().to_string()).expect("record the grant");
        run_component(&path, NetPolicy::Unrestricted).expect("granted run");
        assert_eq!(
            peer.join().expect("echo peer"),
            Some(16),
            "echo peer saw the module's datagram"
        );

        // Same module, same grant, public policy: 127.0.0.1 is not a public
        // destination, so the send is refused and the run fails.
        let refused = run_component(&path, NetPolicy::Public);
        assert!(
            refused.is_err(),
            "the public policy let a datagram go to 127.0.0.1"
        );
    }
}
