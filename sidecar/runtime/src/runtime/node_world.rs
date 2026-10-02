//! `ffrwd:av` 0.19.0: a module that declares its own node shape.
//!
//! Everything a node module is asked goes through here: `describe`,
//! `shape(params, bound)` checked against the refusals the WIT lists, and an
//! instance driven tick by tick with the tick handed over as a borrowed
//! resource. A frame's bytes cross only on `fetch`; a frame handed back with
//! `same` is resolved here to the input's own buffer, so nothing is copied
//! on the way out either.

use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use wasmtime::component::{Component, Resource, ResourceTable};
use wasmtime::Store;
use wasmtime_wasi_http::WasiHttpCtx;

use super::world_0190::node::exports::ffrwd::av::node as wit;
use super::world_0190::node::ffrwd::av::{node_tick, node_types as nt, types as wt};
use super::world_0190::node::NodeModule;
use super::{
    compile, component_exports, conv_0190, egress, engine, granted, has_export, interface, link,
    wasi_ctx, wasm_err, world_0190, Host, Meta, Purpose, StreamInfo, TimeBase, Wants,
};
use crate::node::{
    check_shape, Accepts, Anchor, AudioSpec, BoundStream, Clock, ColorSpec, Emission, Emitted,
    Hold, InputPort, Interval, LikeInput, Node, NodeShape, OutFrame, OutputFormat, OutputPort,
    Pairing, Payload, PortKind, Rational, RowsUse, StreamFormat, Tick, TickStream, VideoSpec,
};

/// The world a node module is built against.
pub const NODE_WORLD: &str = "0.19.0";

/// One tick, entered into the store's resource table for exactly one
/// `process` call, beside the streams `init` bound so `streams` and `info`
/// can answer.
pub struct TickHandle {
    tick: Tick,
    bound: Arc<Vec<BoundStream>>,
}

impl TickHandle {
    fn stream(&self, id: u32) -> wasmtime::Result<&TickStream> {
        if !self.bound.iter().any(|b| b.id == id) {
            return Err(wasmtime::Error::msg(format!(
                "stream {id} was not bound at init"
            )));
        }
        Ok(self.tick.stream(id).unwrap_or(&EMPTY))
    }

    fn bound(&self, id: u32) -> wasmtime::Result<&BoundStream> {
        self.bound
            .iter()
            .find(|b| b.id == id)
            .ok_or_else(|| wasmtime::Error::msg(format!("stream {id} was not bound at init")))
    }
}

static EMPTY: TickStream = TickStream {
    id: 0,
    frames: Vec::new(),
    messages: Vec::new(),
    packets: Vec::new(),
    earlier_rows: Vec::new(),
    feed: None,
    info: None,
    progress: None,
    trailing: Vec::new(),
};

impl nt::Host for Host {}
impl node_tick::Host for Host {}

impl node_tick::HostTick for Host {
    fn pts(&mut self, tick: Resource<TickHandle>) -> wasmtime::Result<i64> {
        Ok(self.table.get(&tick)?.tick.pts)
    }

    fn time_base(&mut self, tick: Resource<TickHandle>) -> wasmtime::Result<wt::Rational> {
        let base = self.table.get(&tick)?.tick.time_base;
        Ok(rational_to_wit(base))
    }

    fn last(&mut self, tick: Resource<TickHandle>) -> wasmtime::Result<bool> {
        Ok(self.table.get(&tick)?.tick.last)
    }

    fn streams(&mut self, tick: Resource<TickHandle>, port: String) -> wasmtime::Result<Vec<u32>> {
        let handle = self.table.get(&tick)?;
        Ok(handle
            .bound
            .iter()
            .filter(|b| b.port == port)
            .map(|b| b.id)
            .collect())
    }

    fn info(&mut self, tick: Resource<TickHandle>, id: u32) -> wasmtime::Result<wt::StreamInfo> {
        let handle = self.table.get(&tick)?;
        let bound = handle.bound(id)?;
        let (info, base) = match &handle.stream(id)?.info {
            Some((info, base)) => (info, *base),
            None => (&bound.info, bound.time_base),
        };
        Ok(stream_info_to_wit(info, base))
    }

    fn feed(&mut self, tick: Resource<TickHandle>, id: u32) -> wasmtime::Result<Option<nt::Feed>> {
        let handle = self.table.get(&tick)?;
        Ok(handle.stream(id)?.feed.as_ref().map(|feed| nt::Feed {
            start: nt::FeedStart {
                tags: feed.start.tags.clone(),
                first_pts: feed.start.first_pts,
                at: feed.start.at,
            },
            ends: feed.ends,
        }))
    }

    fn frames(&mut self, tick: Resource<TickHandle>, id: u32) -> wasmtime::Result<Vec<nt::Frame>> {
        let handle = self.table.get(&tick)?;
        Ok(handle
            .stream(id)?
            .frames
            .iter()
            .enumerate()
            .map(|(index, frame)| nt::Frame {
                pts: frame.pts,
                index: index as u32,
                duration: frame.duration,
                rows: frame.rows.clone(),
            })
            .collect())
    }

    fn fetch(
        &mut self,
        tick: Resource<TickHandle>,
        id: u32,
        index: u32,
    ) -> wasmtime::Result<Vec<u8>> {
        let handle = self.table.get(&tick)?;
        let frames = &handle.stream(id)?.frames;
        let frame = frames.get(index as usize).ok_or_else(|| {
            wasmtime::Error::msg(format!(
                "frame {index} of stream {id} asked of a tick handing {}",
                frames.len()
            ))
        })?;
        Ok(frame.data.as_ref().clone())
    }

    fn messages(
        &mut self,
        tick: Resource<TickHandle>,
        id: u32,
    ) -> wasmtime::Result<Vec<nt::Message>> {
        let handle = self.table.get(&tick)?;
        Ok(handle
            .stream(id)?
            .messages
            .iter()
            .map(|m| nt::Message {
                pts: m.pts,
                data: m.data.clone(),
            })
            .collect())
    }

    fn packets(
        &mut self,
        tick: Resource<TickHandle>,
        id: u32,
    ) -> wasmtime::Result<Vec<wt::Packet>> {
        let handle = self.table.get(&tick)?;
        Ok(handle
            .stream(id)?
            .packets
            .iter()
            .map(|p| wt::Packet {
                pts: p.pts,
                dts: p.dts,
                duration: p.duration,
                keyframe: p.keyframe,
                data: p.data.clone(),
            })
            .collect())
    }

    fn earlier_rows(
        &mut self,
        tick: Resource<TickHandle>,
        id: u32,
    ) -> wasmtime::Result<Vec<nt::TimedRows>> {
        let handle = self.table.get(&tick)?;
        Ok(handle
            .stream(id)?
            .earlier_rows
            .iter()
            .map(|r| nt::TimedRows {
                pts: r.pts,
                rows: r.rows.clone(),
            })
            .collect())
    }

    fn drop(&mut self, tick: Resource<TickHandle>) -> wasmtime::Result<()> {
        // A guest only borrows the tick; the host deletes its own entry once
        // the call is over.
        if tick.owned() {
            self.table.delete(tick)?;
        }
        Ok(())
    }
}

/// Whether the component at `module_path` exports a node.
pub fn exports_node(module_path: &str) -> Result<bool> {
    let component = compile(module_path)?;
    Ok(has_export(&component, &interface("node", NODE_WORLD)))
}

fn instantiate_node(module_path: &str, purpose: Purpose) -> Result<(Store<Host>, NodeModule)> {
    let component = compile(module_path)?;
    check_node_export(&component, module_path)?;
    let (linker, nn, gpu) = link(&component, module_path, purpose)?;
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
            gpu,
        },
    );
    let instance = NodeModule::instantiate(&mut store, &component, &linker)
        .map_err(wasm_err)
        .with_context(|| format!("instantiating {module_path}"))?;
    Ok((store, instance))
}

fn check_node_export(component: &Component, module_path: &str) -> Result<()> {
    let wanted = interface("node", NODE_WORLD);
    if has_export(component, &wanted) {
        return Ok(());
    }
    let exports = component_exports(component);
    if exports.is_empty() {
        bail!("{module_path} exports nothing, so not {wanted}");
    }
    bail!(
        "{module_path} does not export {wanted}; it exports {}",
        exports.join(", ")
    );
}

/// A node module's `describe()`.
pub fn describe_node(module_path: &str) -> Result<Meta> {
    let (mut store, instance) = instantiate_node(module_path, Purpose::Describe)?;
    let m = instance
        .ffrwd_av_node()
        .call_describe(&mut store)
        .map_err(wasm_err)?;
    Ok(world_0190::meta(m))
}

/// A node module's `shape(params, bound)`, checked: the module's own refusal
/// names it, and so does the host's.
pub fn node_shape(module_path: &str, params: &str, bound: &[String]) -> Result<NodeShape> {
    let (mut store, instance) = instantiate_node(module_path, Purpose::Describe)?;
    let name = instance
        .ffrwd_av_node()
        .call_describe(&mut store)
        .map_err(wasm_err)?
        .name;
    shape_of(&mut store, &instance, &name, params, bound)
}

fn shape_of(
    store: &mut Store<Host>,
    instance: &NodeModule,
    name: &str,
    params: &str,
    bound: &[String],
) -> Result<NodeShape> {
    let answered = instance
        .ffrwd_av_node()
        .call_shape(&mut *store, params, bound)
        .map_err(wasm_err)?
        .map_err(|e| anyhow!("{name} refused the shape: {e}"))?;
    let shape = shape_from_wit(answered, name)?;
    check_shape(&shape, bound, name)?;
    Ok(shape)
}

/// One instance of a node module.
pub struct WitNode {
    store: Store<Host>,
    instance: NodeModule,
    meta: Meta,
    shape: NodeShape,
    /// The inputs the call bound, by name, which `set-params` asks the shape
    /// again with.
    bound_names: Vec<String>,
    bound: Arc<Vec<BoundStream>>,
    /// Per output, the last pts a frame or message left at, and the last
    /// dts a packet did, for the checks that neither steps back.
    last: Vec<Option<i64>>,
    finished: bool,
}

impl WitNode {
    /// Instantiates the module, asks its shape for `params` and the inputs
    /// `declared` names (the ones the call bound: a hold input the host
    /// serves from a port is among `bound` and not among them), and opens an
    /// instance on those streams. `latched` names the outputs the query
    /// reads.
    pub fn open(
        module_path: &str,
        params: &str,
        bound: Vec<BoundStream>,
        declared: &[String],
        latched: &[String],
    ) -> Result<WitNode> {
        let (mut store, instance) = instantiate_node(module_path, Purpose::Run)?;
        let meta = world_0190::meta(
            instance
                .ffrwd_av_node()
                .call_describe(&mut store)
                .map_err(wasm_err)?,
        );
        let bound_names: Vec<String> = declared.to_vec();
        let shape = shape_of(&mut store, &instance, &meta.name, params, &bound_names)?;
        check_bound(&shape, &bound, &meta.name)?;
        for wanted in latched {
            if shape.output_index(wanted).is_none() {
                bail!(
                    "{}: output '{wanted}' is not one its shape declares",
                    meta.name
                );
            }
        }
        let wit_bound = bound
            .iter()
            .map(|b| bound_to_wit(b, &meta.name))
            .collect::<Result<Vec<_>>>()?;
        instance
            .ffrwd_av_node()
            .call_init(&mut store, &wit_bound, latched, params)
            .map_err(wasm_err)?
            .map_err(|e| anyhow!("{} refused to open: {e}", meta.name))?;
        let outputs = shape.outputs.len();
        Ok(WitNode {
            store,
            instance,
            meta,
            shape,
            bound_names,
            bound: Arc::new(bound),
            last: vec![None; outputs],
            finished: false,
        })
    }

    pub fn meta(&self) -> &Meta {
        &self.meta
    }
}

/// The streams `init` is handed against the shape: every required input
/// bound, a single input bound once, each stream the kind of its port.
fn check_bound(shape: &NodeShape, bound: &[BoundStream], name: &str) -> Result<()> {
    for (index, stream) in bound.iter().enumerate() {
        if bound[..index].iter().any(|b| b.id == stream.id) {
            bail!("{name}: stream id {} is bound twice", stream.id);
        }
        let Some(port) = shape.input(&stream.port) else {
            bail!("{name} declares no input '{}'", stream.port);
        };
        if stream.format.kind() != port.kind {
            bail!(
                "{name} input '{}' takes {} and is bound a {} stream",
                port.name,
                port.kind.name(),
                stream.format.kind().name()
            );
        }
        crate::node::check_accepts(port, stream, name)?;
    }
    for port in &shape.inputs {
        let count = bound.iter().filter(|b| b.port == port.name).count();
        if port.required && count == 0 {
            bail!(
                "{name} input '{}' is required and this call binds nothing to it",
                port.name
            );
        }
        if !port.many && count > 1 {
            bail!(
                "{name} input '{}' takes one stream and this call binds {count}",
                port.name
            );
        }
    }
    Ok(())
}

impl Node for WitNode {
    fn name(&self) -> &str {
        &self.meta.name
    }

    fn shape(&self) -> &NodeShape {
        &self.shape
    }

    fn set_params(&mut self, params: &str) -> Result<()> {
        let name = self.meta.name.clone();
        let shape = shape_of(
            &mut self.store,
            &self.instance,
            &name,
            params,
            &self.bound_names,
        )?;
        if shape != self.shape {
            bail!("{name} refused params whose shape differs from the instance's");
        }
        self.instance
            .ffrwd_av_node()
            .call_set_params(&mut self.store, params)
            .map_err(wasm_err)?
            .map_err(|e| anyhow!("{name} rejected params: {e}"))
    }

    fn process(&mut self, tick: Tick) -> Result<Emitted> {
        let name = self.meta.name.clone();
        if self.finished {
            bail!("{name}: called again after the final call, which happens once");
        }
        self.finished = tick.last;
        let entry = self
            .store
            .data_mut()
            .table
            .push(TickHandle {
                tick,
                bound: Arc::clone(&self.bound),
            })
            .map_err(|e| anyhow!("entering a tick into the resource table: {e}"))?;
        let handle = Resource::new_borrow(entry.rep());
        let produced = self
            .instance
            .ffrwd_av_node()
            .call_process(&mut self.store, handle)
            .map_err(wasm_err);
        let held = self
            .store
            .data_mut()
            .table
            .delete(entry)
            .map_err(|e| anyhow!("reclaiming a tick from the resource table: {e}"))?;
        let produced = produced?.map_err(|e| anyhow!("{name}: {e}"))?;
        self.emitted(produced, &held.tick)
    }
}

impl WitNode {
    /// What a call returned, in the host's spelling and checked: each item on
    /// a port the shape declares and of its kind, `same` resolved to the
    /// input's buffer, and no port stepping back in time.
    fn emitted(&mut self, produced: wit::Emitted, tick: &Tick) -> Result<Emitted> {
        let name = &self.meta.name;
        let mut items = Vec::with_capacity(produced.items.len());
        for emission in produced.items {
            let Some(port) = self.shape.output_index(&emission.port) else {
                bail!(
                    "{name} emitted on '{}', which is no output of its shape",
                    emission.port
                );
            };
            let kind = self.shape.outputs[port].kind;
            let (payload, time) = match emission.payload {
                wit::Payload::Frame(f) => {
                    check_kind(name, &emission.port, kind, "a frame", kind.is_frames())?;
                    (
                        Payload::Frame(OutFrame {
                            pts: f.pts,
                            duration: f.duration,
                            data: Arc::new(f.data),
                            rows: Vec::new(),
                        }),
                        Some(f.pts),
                    )
                }
                wit::Payload::Same(same) => {
                    check_kind(name, &emission.port, kind, "a frame", kind.is_frames())?;
                    let data = self.same(port, &same, tick)?;
                    (
                        Payload::Frame(OutFrame {
                            pts: same.pts,
                            duration: same.duration,
                            data,
                            rows: Vec::new(),
                        }),
                        Some(same.pts),
                    )
                }
                wit::Payload::Message(m) => {
                    check_kind(
                        name,
                        &emission.port,
                        kind,
                        "a message",
                        kind == PortKind::Data,
                    )?;
                    (
                        Payload::Message(super::Message {
                            pts: m.pts,
                            data: m.data,
                        }),
                        Some(m.pts),
                    )
                }
                wit::Payload::Packet(p) => {
                    check_kind(
                        name,
                        &emission.port,
                        kind,
                        "a packet",
                        kind == PortKind::Packets,
                    )?;
                    (
                        Payload::Packet(super::Packet {
                            pts: p.pts,
                            dts: p.dts,
                            duration: p.duration,
                            keyframe: p.keyframe,
                            data: p.data,
                        }),
                        p.dts,
                    )
                }
            };
            if let Some(time) = time {
                if let Some(before) = self.last[port] {
                    if time < before {
                        bail!(
                            "{name} emitted on '{}' at {time} after {before}; a port never steps                              back",
                            emission.port
                        );
                    }
                }
                self.last[port] = Some(time);
            }
            items.push(Emission { port, payload });
        }
        Ok(Emitted {
            items,
            rows: produced.rows,
            finished: produced.finished,
        })
    }

    /// The buffer behind a `same`: frame `index` of stream `id` this tick,
    /// in the port's own format.
    fn same(&self, port: usize, same: &wit::SameFrame, tick: &Tick) -> Result<Arc<Vec<u8>>> {
        let name = &self.meta.name;
        let output = &self.shape.outputs[port].name;
        let Some(bound) = self.bound.iter().find(|b| b.id == same.id) else {
            bail!(
                "{name} handed back stream {} on '{output}', which was not bound",
                same.id
            );
        };
        let frames = tick
            .stream(same.id)
            .map(|s| s.frames.as_slice())
            .unwrap_or(&[]);
        let Some(frame) = frames.get(same.index as usize) else {
            bail!(
                "{name} handed back frame {} of stream {} on '{output}', and the tick handed {}",
                same.index,
                same.id,
                frames.len()
            );
        };
        if let StreamFormat::Audio(_) = bound.format {
            let input = self.shape.input(&bound.port);
            if input.is_some_and(|p| p.stride != p.window) {
                bail!(
                    "{name} handed back audio from '{}' on '{output}', whose windows overlap; \
                     every sample would leave more than once",
                    bound.port
                );
            }
        }
        if let Some(wanted) = self.output_format(port) {
            if !same_format(&wanted, &bound.format) {
                bail!(
                    "{name} handed back a frame of '{}' on '{output}', whose format is not the \
                     port's",
                    bound.port
                );
            }
        }
        Ok(Arc::clone(&frame.data))
    }

    /// An output's format as the shape and the bound streams settle it.
    fn output_format(&self, port: usize) -> Option<FormatOf> {
        output_format(&self.shape, &self.bound, port)
    }
}

/// An output's frames, as far as a `same` needs to know them.
#[derive(Debug, PartialEq)]
pub enum FormatOf {
    Video {
        width: u32,
        height: u32,
        pix_fmt: String,
    },
    Audio {
        sample_rate: u32,
        channels: u32,
        sample_fmt: String,
    },
}

fn format_of(format: &StreamFormat) -> Option<FormatOf> {
    match format {
        StreamFormat::Video(v) => Some(FormatOf::Video {
            width: v.width,
            height: v.height,
            pix_fmt: v.pix_fmt.to_string(),
        }),
        StreamFormat::Audio(a) => Some(FormatOf::Audio {
            sample_rate: a.sample_rate,
            channels: a.channels,
            sample_fmt: a.sample_fmt.to_string(),
        }),
        _ => None,
    }
}

fn same_format(wanted: &FormatOf, arriving: &StreamFormat) -> bool {
    format_of(arriving).as_ref() == Some(wanted)
}

/// An output port's frame format: what it declares, the clock input's where
/// it declares none, or the input it follows with its override.
pub fn output_format(shape: &NodeShape, bound: &[BoundStream], port: usize) -> Option<FormatOf> {
    let first_of = |input: &str| bound.iter().find(|b| b.port == input).map(|b| &b.format);
    match &shape.outputs[port].format {
        Some(OutputFormat::Video(v)) => Some(FormatOf::Video {
            width: v.width,
            height: v.height,
            pix_fmt: v.pix_fmt.clone(),
        }),
        Some(OutputFormat::Audio(a)) => Some(FormatOf::Audio {
            sample_rate: a.sample_rate,
            channels: a.channels,
            sample_fmt: a.sample_fmt.clone(),
        }),
        Some(OutputFormat::Like(like)) => {
            let mut format = format_of(first_of(&like.port)?)?;
            match &mut format {
                FormatOf::Video { pix_fmt, .. } => {
                    if let Some(wanted) = &like.pixel_format {
                        *pix_fmt = wanted.clone();
                    }
                }
                FormatOf::Audio { sample_fmt, .. } => {
                    if let Some(wanted) = &like.sample_format {
                        *sample_fmt = wanted.clone();
                    }
                }
            }
            Some(format)
        }
        None => match &shape.clock {
            Clock::Input(clock) => format_of(first_of(clock)?),
            _ => None,
        },
        Some(OutputFormat::Data(_)) | Some(OutputFormat::Packets(_)) => None,
    }
}

fn check_kind(name: &str, port: &str, kind: PortKind, what: &str, fits: bool) -> Result<()> {
    if !fits {
        bail!(
            "{name} emitted {what} on '{port}', which carries {}",
            kind.name()
        );
    }
    Ok(())
}

fn rational_to_wit(base: TimeBase) -> wt::Rational {
    wt::Rational {
        num: i32::try_from(base.num).unwrap_or(i32::MAX),
        den: i32::try_from(base.den).unwrap_or(i32::MAX),
    }
}

fn stream_info_to_wit(info: &StreamInfo, base: TimeBase) -> wt::StreamInfo {
    wt::StreamInfo {
        index: info.index,
        kind: info.kind.clone(),
        codec: info.codec.clone(),
        duration: info.duration,
        tags: info.tags.clone(),
        time_base: rational_to_wit(base),
    }
}

fn bound_to_wit(stream: &BoundStream, name: &str) -> Result<nt::BoundStream> {
    let format = match &stream.format {
        StreamFormat::Video(v) => nt::OutputFormat::Video(wt::VideoFormat {
            width: v.width,
            height: v.height,
            pix_fmt: v.pix_fmt.to_string(),
            color: v.color.map(world_0190::color_info),
        }),
        StreamFormat::Audio(a) => nt::OutputFormat::Audio(wt::AudioFormat {
            sample_rate: a.sample_rate,
            channels: a.channels,
            sample_fmt: a.sample_fmt.to_string(),
            channel_layout: a.channel_layout.map(str::to_string),
        }),
        StreamFormat::Data(codec) => nt::OutputFormat::Data(codec.clone()),
        StreamFormat::Packets(coded) => {
            nt::OutputFormat::Packets(conv_0190::coded_stream_to_wit(coded, name)?)
        }
    };
    Ok(nt::BoundStream {
        port: stream.port.clone(),
        id: stream.id,
        info: stream_info_to_wit(&stream.info, stream.time_base),
        format: Some(format),
        rendition: world_0190::rendition_meta(stream.rendition.clone()),
        row: stream.row,
        decode_delay: stream.decode_delay,
        latency: stream.latency,
    })
}

fn kind_from_wit(kind: nt::PortKind) -> PortKind {
    match kind {
        nt::PortKind::Video => PortKind::Video,
        nt::PortKind::Audio => PortKind::Audio,
        nt::PortKind::Data => PortKind::Data,
        nt::PortKind::Packets => PortKind::Packets,
    }
}

fn rational_from_wit(r: wt::Rational) -> Rational {
    Rational {
        num: r.num,
        den: r.den,
    }
}

fn color_from_wit(c: wt::ColorInfo) -> ColorSpec {
    ColorSpec {
        range: c.range,
        primaries: c.primaries,
        trc: c.trc,
        space: c.space,
    }
}

fn shape_from_wit(shape: nt::NodeShape, name: &str) -> Result<NodeShape> {
    let mut outputs = Vec::with_capacity(shape.outputs.len());
    for p in shape.outputs {
        outputs.push(OutputPort {
            name: p.name,
            kind: kind_from_wit(p.kind),
            format: match p.format {
                None => None,
                Some(f) => Some(output_format_from_wit(f, name)?),
            },
            time_base: p.time_base.map(rational_from_wit),
            latency: p.latency,
            schema: p.schema,
            row: p.row,
        });
    }
    Ok(NodeShape {
        inputs: shape
            .inputs
            .into_iter()
            .map(|p| InputPort {
                name: p.name,
                kind: kind_from_wit(p.kind),
                required: p.required,
                many: p.many,
                pairing: match p.pairing {
                    nt::Pairing::Lockstep => Pairing::Lockstep,
                    nt::Pairing::Hold(h) => Pairing::Hold(Hold {
                        anchor: match h.anchor {
                            nt::Anchor::SharedClock => Anchor::SharedClock,
                            nt::Anchor::FirstFrame => Anchor::FirstFrame,
                            nt::Anchor::Tagged(tag) => Anchor::Tagged(tag),
                        },
                        lead: h.lead,
                        linger: h.linger,
                        timeout: h.timeout,
                        group: h.group,
                        port_param: h.port_param,
                    }),
                    nt::Pairing::Interval(i) => Pairing::Interval(Interval {
                        latency: i.latency,
                        ahead: i.ahead,
                    }),
                    nt::Pairing::Arrival => Pairing::Arrival,
                },
                rows: match p.rows {
                    nt::RowsUse::Ignore => RowsUse::Ignore,
                    nt::RowsUse::PerFrame => RowsUse::PerFrame,
                    nt::RowsUse::State => RowsUse::State,
                },
                window: p.window,
                stride: p.stride,
                accepts: Accepts {
                    pixel_formats: p.accepts.pixel_formats,
                    sample_formats: p.accepts.sample_formats,
                    sample_rates: p.accepts.sample_rates,
                    channel_counts: p.accepts.channel_counts,
                    codecs: p.accepts.codecs,
                    wants: match p.accepts.wants {
                        wt::Wants::All => Wants::All,
                        wt::Wants::Keyframes => Wants::Keyframes,
                        wt::Wants::First => Wants::First,
                    },
                    like: p.accepts.like,
                },
                schema: p.schema,
            })
            .collect(),
        outputs,
        clock: match shape.clock {
            nt::Clock::Input(port) => Clock::Input(port),
            nt::Clock::Rate(rate) => Clock::Rate(rational_from_wit(rate)),
            nt::Clock::RateOf(port) => Clock::RateOf(port),
            nt::Clock::SelfClocked => Clock::SelfClocked,
        },
        pure: shape.pure,
        one_to_one: shape.one_to_one,
        bounded: shape.bounded,
        relation: shape.relation,
    })
}

fn output_format_from_wit(format: nt::OutputFormat, name: &str) -> Result<OutputFormat> {
    Ok(match format {
        nt::OutputFormat::Video(v) => OutputFormat::Video(VideoSpec {
            width: v.width,
            height: v.height,
            pix_fmt: v.pix_fmt,
            color: v.color.map(color_from_wit),
        }),
        nt::OutputFormat::Audio(a) => OutputFormat::Audio(AudioSpec {
            sample_rate: a.sample_rate,
            channels: a.channels,
            sample_fmt: a.sample_fmt,
            channel_layout: a.channel_layout,
        }),
        nt::OutputFormat::Data(codec) => OutputFormat::Data(codec),
        nt::OutputFormat::Packets(c) => {
            OutputFormat::Packets(conv_0190::coded_stream_from_wit(c, name)?)
        }
        nt::OutputFormat::Like(l) => OutputFormat::Like(LikeInput {
            port: l.port,
            pixel_format: l.pixel_format,
            sample_format: l.sample_format,
        }),
    })
}
