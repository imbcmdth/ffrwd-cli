//! A network of older frame modules, run on the node lanes.
//!
//! Every node is a lane whose calls are cut the way its world's host cut
//! them: a window as soon as it is whole, every further pad one frame at the
//! clock's pts, and a last call with what the strides left over and the rows
//! that reached the first pad with no frame to ride. Rows ride their frames
//! from lane to lane, and each output is written one call at a time by the
//! sink its world's host wrote it with. The host's own `rowfilter`,
//! `rowmerge` and `leaky` are lanes like any other.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::thread;

use anyhow::{bail, Context, Result};
use ffrwd_wasm_runtime::node::{
    Clock, Emission, Emitted, Node, NodeShape, OutFrame, Pairing, Payload, PortKind, RowsUse,
    TickFrame,
};
use ffrwd_wasm_runtime::runtime::{Format, Frame, Media, Processed, Shape, StreamInfo};

use crate::adapters::{self, FilterNode};
use crate::edges;
use crate::lanes::{self, Consumer, Intake, LaneSpec, Opener, Plan, PortOut, Scheduler};
use crate::leaky::Leaky;
use crate::network::{Network, Source};
use crate::rowfilter::RowFilter;
use crate::rowmerge::RowMerge;
use crate::tick::{Item, Lockstep};
use crate::{check_frame, Input, Sink};

/// How to open one more instance of a lane's module, for a pure lane growing
/// under pressure.
pub struct Reopen {
    pub path: String,
    pub params: String,
    pub format: Format,
    pub info: StreamInfo,
}

/// What executes one lane's calls: a module instance, driven as a node
/// whatever world it was built against, or one of the host's own.
pub enum Runner {
    Node(Box<dyn Node>),
    Rows(RowFilter),
    Merge(RowMerge),
    Leaky(Box<Leaky>),
}

/// One node of the opened network, as the lanes take it over.
pub struct LaneSeed {
    pub name: String,
    /// The instances opened so far; ordinarily one.
    pub runners: Vec<Runner>,
    pub shape: Shape,
    pub sources: Vec<Source>,
    pub format: Format,
    /// Set only for a lane that may grow more instances.
    pub reopen: Option<Reopen>,
}

/// One of the host's own frame nodes: frames pass, their rows filtered,
/// merged or the late pictures dropped.
struct HostFrames {
    runner: Runner,
    shape: NodeShape,
    name: String,
}

impl Node for HostFrames {
    fn name(&self) -> &str {
        &self.name
    }

    fn shape(&self) -> &NodeShape {
        &self.shape
    }

    fn host_progress(&self) -> bool {
        false
    }

    fn set_params(&mut self, _params: &str) -> Result<()> {
        bail!(
            "{} takes its options from the command line and no params",
            self.name
        )
    }

    fn process(&mut self, tick: ffrwd_wasm_runtime::node::Tick) -> Result<Emitted> {
        let last = tick.last;
        let mut first = tick.streams.into_iter().next().unwrap_or_default();
        if let Runner::Leaky(leaky) = &mut self.runner {
            if !first.packets.is_empty() {
                let items = std::mem::take(&mut first.packets)
                    .into_iter()
                    .filter_map(|p| leaky.pass_packet(p))
                    .map(|p| Emission {
                        port: 0,
                        payload: Payload::Packet(p),
                    })
                    .collect();
                if last {
                    leaky.finish();
                }
                return Ok(Emitted {
                    items,
                    rows: Vec::new(),
                    finished: false,
                });
            }
        }
        let frames = first.frames.into_iter().map(|f| Frame {
            pts: f.pts,
            data: f.data,
            rows: f.rows,
        });
        let trailing = first.trailing;
        let processed = match &mut self.runner {
            Runner::Rows(rows) => Processed {
                frames: frames.map(|f| rows.pass(f)).collect(),
                trailing: if last {
                    rows.keep(trailing)
                } else {
                    Vec::new()
                },
            },
            Runner::Merge(merge) => {
                let frames = frames.map(|f| merge.pass(f)).collect();
                // The run still open when the rows run out has no frame left
                // to ride, so it leaves with the trailing ones.
                let mut trailing = if last {
                    merge.merged(trailing)
                } else {
                    Vec::new()
                };
                if last {
                    trailing.extend(merge.finish());
                }
                Processed { frames, trailing }
            }
            Runner::Leaky(leaky) => {
                let frames = frames.filter_map(|f| leaky.pass(f)).collect();
                if last {
                    leaky.finish();
                }
                Processed {
                    frames,
                    trailing: if last { trailing } else { Vec::new() },
                }
            }
            Runner::Node(_) => unreachable!("a module lane runs its node"),
        };
        let mut items: Vec<Emission> = processed
            .frames
            .into_iter()
            .map(|f| Emission {
                port: 0,
                payload: Payload::Frame(OutFrame {
                    pts: f.pts,
                    duration: None,
                    data: f.data,
                    rows: f.rows,
                }),
            })
            .collect();
        if !processed.trailing.is_empty() {
            items.push(Emission {
                port: 0,
                payload: Payload::Rows(processed.trailing),
            });
        }
        Ok(Emitted {
            items,
            rows: Vec::new(),
            finished: false,
        })
    }
}

/// `leaky` as a node of a node network: one picture input, `in0`, and the
/// pictures that were not too late on `out`.
pub(crate) fn leaky_node(name: &str, leaky: Leaky, kind: PortKind) -> Box<dyn Node> {
    let shape = NodeShape {
        inputs: vec![adapters::input(
            "in0",
            kind,
            true,
            false,
            Pairing::Lockstep,
            RowsUse::PerFrame,
        )],
        outputs: vec![adapters::output("out", kind, None)],
        clock: Clock::Input("in0".into()),
        pure: false,
        one_to_one: false,
        bounded: true,
        relation: Vec::new(),
    };
    Box::new(HostFrames {
        runner: Runner::Leaky(Box::new(leaky)),
        shape,
        name: name.to_string(),
    })
}

/// The shape a lane of `seed` is scheduled by: one port per pad, the first
/// the clock, one output. Only a module over video spreads across workers:
/// an audio module's output is one continuous run of samples per instance.
fn lane_shape(seed: &LaneSeed, module: bool) -> NodeShape {
    let kind = match seed.format.media {
        Media::Video(_) => PortKind::Video,
        Media::Audio(_) => PortKind::Audio,
    };
    let inputs = (0..seed.sources.len())
        .map(|pad| {
            let mut port = adapters::input(
                &format!("in{pad}"),
                kind,
                true,
                false,
                Pairing::Lockstep,
                RowsUse::PerFrame,
            );
            if pad == 0 {
                port.window = seed.shape.window;
                port.stride = seed.shape.stride;
            }
            port
        })
        .collect();
    NodeShape {
        inputs,
        outputs: vec![adapters::output("out", kind, None)],
        clock: Clock::Input("in0".into()),
        pure: module && seed.shape.pure && seed.format.video().is_some(),
        one_to_one: seed.shape.one_to_one,
        bounded: true,
        relation: Vec::new(),
    }
}

/// Runs `net` over `readers`, writing `sinks`, until every lane has made its
/// last call or no output takes anything more.
pub fn run(
    net: Network,
    readers: Vec<Input>,
    formats: &[Format],
    sinks: Vec<Sink>,
    jobs: Option<usize>,
) -> Result<()> {
    let inputs = readers.len();
    let seeds = net.into_seeds();
    let node_stream = |lane: usize| (inputs + lane) as u32;
    let mut consumers: HashMap<u32, Vec<Consumer>> = HashMap::new();
    let mut specs: Vec<LaneSpec> = Vec::with_capacity(seeds.len());
    for (index, seed) in seeds.into_iter().enumerate() {
        let pads: Vec<u32> = seed
            .sources
            .iter()
            .map(|source| match source {
                Source::Input(input) => *input as u32,
                Source::Node(lane) => node_stream(*lane),
            })
            .collect();
        let mut read: Vec<u32> = pads.clone();
        read.sort_unstable();
        read.dedup();
        for id in read {
            consumers.entry(id).or_default().push(Consumer::Lane(index));
        }
        let module = matches!(seed.runners.first(), Some(Runner::Node(_)));
        let unbuffered = matches!(seed.runners.first(), Some(Runner::Leaky(_)));
        let shape = lane_shape(&seed, module);
        let opener: Option<Opener> = seed.reopen.map(|reopen| {
            let opener: Opener = Arc::new(move || {
                let node =
                    FilterNode::open(&reopen.path, &reopen.format, &reopen.info, &reopen.params)
                        .with_context(|| format!("opening another instance of {}", reopen.path))?;
                Ok(Box::new(node) as Box<dyn Node>)
            });
            opener
        });
        let runners: Vec<Box<dyn Node>> = seed
            .runners
            .into_iter()
            .map(|runner| match runner {
                Runner::Node(node) => node,
                host => Box::new(HostFrames {
                    runner: host,
                    shape: shape.clone(),
                    name: seed.name.clone(),
                }) as Box<dyn Node>,
            })
            .collect();
        let lockstep = Lockstep::new(seed.shape, &seed.format, pads.len());
        specs.push(LaneSpec {
            name: seed.name,
            intake: Intake::Windows {
                lockstep,
                pads: pads.clone(),
            },
            bound: pads,
            state: Vec::new(),
            tick_base: seed.format.time_base,
            runners,
            opener,
            ports: vec![None],
            rows: None,
            unbuffered,
            shape,
        });
    }

    let wake_slot: Arc<OnceLock<Arc<dyn Fn() + Send + Sync>>> = Arc::new(OnceLock::new());
    let wake: Arc<dyn Fn() + Send + Sync> = {
        let slot = Arc::clone(&wake_slot);
        Arc::new(move || {
            if let Some(wake) = slot.get() {
                wake();
            }
        })
    };
    let mut writers = Vec::with_capacity(sinks.len());
    for (w, sink) in sinks.into_iter().enumerate() {
        let lane = sink.node;
        consumers
            .entry(node_stream(lane))
            .or_default()
            .push(Consumer::Writer(w, 0));
        let name = specs[lane].name.clone();
        writers.push(edges::spawn_sink(sink, name, Arc::clone(&wake)));
    }
    for (index, spec) in specs.iter_mut().enumerate() {
        let stream = node_stream(index);
        if consumers.contains_key(&stream) {
            spec.ports[0] = Some(PortOut {
                stream,
                base: spec.tick_base,
                latency: 0.0,
            });
        }
    }

    let plan = Plan {
        lanes: specs,
        consumers,
        writers: writers.iter().map(|w| Arc::clone(&w.queue)).collect(),
        mirrors: HashMap::new(),
    };
    let scheduler = Arc::new(Scheduler::start(plan, lanes::worker_count(jobs))?);
    let _ = wake_slot.set(scheduler.waker());

    let mut feeding = Vec::with_capacity(inputs);
    for (input, mut reader) in readers.into_iter().enumerate() {
        let scheduler = Arc::clone(&scheduler);
        let format = formats[input];
        feeding.push(thread::spawn(move || {
            if let Err(e) = feed(&mut reader, input, &format, &scheduler) {
                scheduler.fail(e);
            }
        }));
    }
    let finished = scheduler.finish();
    drop(feeding);
    finished?;
    for writer in &mut writers {
        writer.join();
    }
    for writer in &writers {
        if let Some(failure) = writer.queue.failure() {
            bail!("{failure}");
        }
    }
    Ok(())
}

/// Every frame of input `input`, with the rows riding it, and at its end the
/// rows that rode nothing.
fn feed(reader: &mut Input, input: usize, format: &Format, scheduler: &Scheduler) -> Result<()> {
    let id = input as u32;
    let mut buf: Vec<u8> = Vec::new();
    let mut index = 0u64;
    loop {
        let read = reader
            .read_frame(&mut buf)
            .with_context(|| format!("reading frame {index} of input {input}"))?;
        let Some(pts) = read else {
            let trailing = reader.take_trailing();
            scheduler.end_with(id, &trailing);
            return Ok(());
        };
        check_frame(format, &buf, index)?;
        index += 1;
        let frame = TickFrame {
            pts,
            duration: None,
            data: Arc::new(std::mem::take(&mut buf)),
            rows: reader.take_rows(pts),
        };
        if !scheduler.arrive(id, Item::Frame(frame)) {
            return Ok(());
        }
    }
}
