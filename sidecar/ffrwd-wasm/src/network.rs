//! A network of modules in one process: the DAG the `-filter_complex` string
//! describes, opened and driven.
//!
//! Every node buffers the window its module asked for and is walked in
//! topological order once per batch of arriving frames. One module's output
//! frames are handed to every reader in memory - rows riding inside the frame,
//! so a reader that acts on them gets them without a wire. The final call
//! cascades the same way: a node's last window is made, and whatever it
//! produced flows on through the nodes after it.
//!
//! A chain of one module is a network of one, which is how the single-module
//! spelling stays the same code.

use std::collections::{HashMap, VecDeque};

use anyhow::{anyhow, bail, Context, Result};
use ffrwd_wasm_runtime::runtime::{
    self, Described, Filter, Format, Frame, Kind, Media, Processed, Shape, StreamInfo,
};

use crate::graph::{EdgeKind, Pad, ParsedNode};
use crate::rowfilter::{self, RowFilter};
use crate::windows::Windows;

/// What drives one node: a module, or the one node the host answers for
/// itself.
enum Unit {
    // Boxed because a module instance carries a whole wasm store, and every
    // node would otherwise be that size.
    Module(Box<Filter>),
    Rows(RowFilter),
}

impl Unit {
    fn name(&self) -> &str {
        match self {
            Unit::Module(filter) => filter.name(),
            Unit::Rows(_) => rowfilter::NODE,
        }
    }

    fn shape(&self) -> Shape {
        match self {
            Unit::Module(filter) => filter.shape(),
            Unit::Rows(_) => rowfilter::SHAPE,
        }
    }

    /// Whether upstream rows may leave on this node's own output frames. The
    /// rows node passes on the ones it kept, so they may.
    fn forwards_rows(&self) -> bool {
        match self {
            Unit::Module(filter) => filter.forwards_rows(),
            Unit::Rows(_) => true,
        }
    }

    /// Whether this node acts on the rows arriving with its frames. Acting on
    /// them is all the rows node does.
    fn reads_rows(&self) -> bool {
        match self {
            Unit::Module(filter) => filter.reads_rows(),
            Unit::Rows(_) => true,
        }
    }
}

/// Where one of a node's streams comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    /// A `-i` input, by index.
    Input(usize),
    /// Another node's output, by index in topological order.
    Node(usize),
}

/// One module in the network: its instance, the window it is buffering
/// toward, and one queue per stream it reads.
struct Node {
    unit: Unit,
    /// The node's own name, for a message a borrow of `unit` cannot reach.
    name: String,
    /// Whether a frame's incoming rows can leave this module still attached,
    /// as the module itself declares.
    forwards_rows: bool,
    /// Whether the module acts on the rows arriving with its frames.
    reads_rows: bool,
    /// The window this node buffers toward. A module reading several streams
    /// is window 1, stride 1, so its pads go straight to it and this is never
    /// fed.
    windows: Windows,
    sources: Vec<Source>,
    /// One buffer per pad, in pad order. A pad's frames wait here until every
    /// other pad has one at the same timestamp.
    queues: Vec<VecDeque<Frame>>,
    /// Rows the modules feeding this one had no frame to put them on, for
    /// this one's final call.
    trailing: Vec<String>,
}

impl Node {
    fn new(unit: Unit, sources: Vec<Source>, format: &Format) -> Node {
        let windows = Windows::new(unit.shape(), format);
        let queues = sources.iter().map(|_| VecDeque::new()).collect();
        let name = unit.name().to_string();
        let forwards_rows = unit.forwards_rows();
        let reads_rows = unit.reads_rows();
        Node {
            unit,
            name,
            forwards_rows,
            reads_rows,
            windows,
            sources,
            queues,
            trailing: Vec::new(),
        }
    }

    /// Queues what each pad received, then runs everything those frames
    /// completed.
    fn feed(&mut self, arrived: Vec<Vec<Frame>>) -> Result<Vec<Frame>> {
        for (queue, frames) in self.queues.iter_mut().zip(arrived) {
            queue.extend(frames);
        }
        let mut out = Vec::new();
        while self.queues.iter().all(|q| !q.is_empty()) {
            let pads = self.take()?;
            out.extend(self.run(pads)?);
        }
        Ok(out)
    }

    /// One frame off every pad, at one timestamp. The pads are read in
    /// lockstep: every head must carry the timestamp pad 0's head carries, and
    /// a pad that skipped it is a refusal rather than a frame quietly dropped.
    ///
    /// Rows ride pad 0. What arrived on any other pad is dropped here, so a
    /// module is handed rows on its first pad and nowhere else.
    fn take(&mut self) -> Result<Vec<Frame>> {
        let head = self.queues[0]
            .front()
            .map(|f| f.pts)
            .ok_or_else(|| anyhow!("{}: a queue emptied mid-window", self.name))?;
        for (pad, queue) in self.queues.iter().enumerate().skip(1) {
            let other = queue.front().map(|f| f.pts).unwrap_or(head);
            if other != head {
                bail!(
                    "{}: pad 0 is at pts {head} and pad {pad} at pts {other}; a module reading \
                     several streams reads them in lockstep, one frame per pad at one timestamp",
                    self.name
                );
            }
        }
        let mut pads = Vec::with_capacity(self.queues.len());
        for (pad, queue) in self.queues.iter_mut().enumerate() {
            let mut frame = queue.pop_front().expect("every head matched pts");
            if pad > 0 {
                frame.rows.clear();
            }
            pads.push(frame);
        }
        Ok(pads)
    }

    /// One timestamp's pads through the module. A module reading one stream
    /// buffers toward the window it asked for; one reading several is window
    /// 1, stride 1, so the pads are the call.
    fn run(&mut self, mut pads: Vec<Frame>) -> Result<Vec<Frame>> {
        // The rows node reads one stream at window 1, stride 1, and never
        // looks at the pixels, so the frame moves straight through rather
        // than being buffered and copied.
        if let Unit::Rows(rows) = &mut self.unit {
            return Ok(pads.into_iter().map(|frame| rows.pass(frame)).collect());
        }
        let Unit::Module(filter) = &mut self.unit else {
            unreachable!("the rows node was taken above")
        };
        if pads.len() > 1 {
            return Ok(filter.process_window(&pads, &[], false)?.frames);
        }
        let frame = pads.pop().expect("a node reads at least one stream");
        let mut out = Vec::new();
        for window in self.windows.push(frame, &self.name)? {
            out.extend(filter.process_window(&window, &[], false)?.frames);
        }
        Ok(out)
    }

    /// The final call, carrying whatever the last stride left over and
    /// whatever rows reached this module with no frame to ride.
    fn finish(&mut self) -> Result<Processed> {
        for (pad, queue) in self.queues.iter().enumerate() {
            if !queue.is_empty() {
                bail!(
                    "{}: pad {pad} ends with {} frame(s) that never paired with the other pads; \
                     a module reading several streams reads them in lockstep",
                    self.name,
                    queue.len()
                );
            }
        }
        let tail = self.windows.tail();
        let trailing = std::mem::take(&mut self.trailing);
        match &mut self.unit {
            // The same predicate judges the rows that never found a frame.
            Unit::Rows(rows) => Ok(Processed {
                frames: Vec::new(),
                trailing: rows.keep(trailing),
            }),
            Unit::Module(filter) => filter.process_window(&tail, &trailing, true),
        }
    }
}

/// One module bound to a name by `-m name=path`.
#[derive(Debug, Clone)]
pub struct Binding {
    pub name: String,
    pub path: String,
}

/// What the final call over the whole network produced, per node.
pub struct Drained {
    pub frames: Vec<Vec<Frame>>,
    pub trailing: Vec<Vec<String>>,
}

/// The opened network: nodes in topological order, the label each one writes,
/// and which input each descends from.
pub struct Network {
    nodes: Vec<Node>,
    labels: HashMap<String, usize>,
    roots: Vec<usize>,
    /// The last node to read each stream, which is the one that may take the
    /// frames rather than copy them. A node an output is mapped to is never in
    /// here: its frames are still wanted after the walk.
    last_reader: HashMap<Source, usize>,
}

impl Network {
    /// Opens every module of `parsed`, checking the graph first: every module
    /// name bound, every label written once and read at least once, every
    /// input index backed by a `-i`.
    ///
    /// `mapped` is the `-map` target of each output, which is what makes a
    /// label written by the last node consumed rather than dangling.
    pub fn open(
        parsed: &[ParsedNode],
        bindings: &[Binding],
        formats: &[Format],
        streams: &[StreamInfo],
        mapped: &[String],
    ) -> Result<Network> {
        let paths = bound_paths(bindings)?;
        let writer = writers(parsed)?;
        for target in mapped {
            if !writer.contains_key(target.as_str()) {
                bail!("-map [{target}] names a label the network does not write");
            }
        }
        check_consumed(parsed, mapped, &writer)?;

        let sources = resolve(parsed, &writer, formats.len())?;
        check_inputs_read(&sources, formats.len())?;
        let order = topological(parsed, &sources)?;
        let position: HashMap<usize, usize> = order
            .iter()
            .enumerate()
            .map(|(place, node)| (*node, place))
            .collect();

        check_input_kinds(parsed, formats)?;

        let mut schemas: HashMap<String, serde_json::Value> = HashMap::new();
        let mut nodes = Vec::with_capacity(order.len());
        let mut roots = Vec::with_capacity(order.len());
        let mut labels = HashMap::new();

        for original in &order {
            let node = &parsed[*original];

            // Renumbered onto the topological order the nodes are driven in.
            let placed: Vec<Source> = sources[*original]
                .iter()
                .map(|source| match source {
                    Source::Input(index) => Source::Input(*index),
                    Source::Node(index) => Source::Node(position[index]),
                })
                .collect();
            let root = root_input(&placed, &roots);
            check_one_format(&node.module, &placed, &roots, formats)?;

            // The rows node is the host's own: nothing is compiled, and it
            // carries whichever kind reaches it, so neither the module kind
            // nor a params schema applies.
            let unit = if node.module == rowfilter::NODE {
                check_pad_count(&node.module, 1, placed.len())?;
                Unit::Rows(RowFilter::open(&node.options)?)
            } else {
                let path = paths.get(node.module.as_str()).ok_or_else(|| {
                    anyhow!(
                        "the network names module '{}', which no -m binds; bound: {}",
                        node.module,
                        bound_names(bindings)
                    )
                })?;
                let described = runtime::describe(path)
                    .with_context(|| format!("describing module '{}'", node.module))?;
                check_module_kind(node, &described, formats[root].kind())?;
                check_pad_count(&node.module, described.inputs as usize, placed.len())?;

                let schema = match schemas.get(path.as_str()) {
                    Some(schema) => schema.clone(),
                    None => {
                        let schema = parse_schema(&described.meta.params_schema, &node.module)?;
                        schemas.insert(path.clone(), schema.clone());
                        schema
                    }
                };
                let params = params_json(&node.module, &schema, &node.options)?;
                let filter = Filter::open(path, &formats[root], &streams[root], &params)
                    .with_context(|| format!("opening module '{}' from {path}", node.module))?;
                Unit::Module(Box::new(filter))
            };

            for label in &node.outputs {
                labels.insert(label.clone(), nodes.len());
            }
            roots.push(root);
            nodes.push(Node::new(unit, placed, &formats[root]));
        }

        let sunk: Vec<usize> = mapped
            .iter()
            .filter_map(|t| labels.get(t).copied())
            .collect();
        let last_reader = last_readers(&nodes, &sunk);

        Ok(Network {
            nodes,
            labels,
            roots,
            last_reader,
        })
    }

    /// A network of one module, wired from one input to one node - the shape
    /// the single-module spelling runs as.
    pub fn single(filter: Filter, format: &Format) -> Network {
        Network {
            nodes: vec![Node::new(
                Unit::Module(Box::new(filter)),
                vec![Source::Input(0)],
                format,
            )],
            labels: HashMap::new(),
            roots: vec![0],
            last_reader: HashMap::from([(Source::Input(0), 0)]),
        }
    }

    /// The node a `-map` target names.
    pub fn node_for(&self, label: &str) -> Option<usize> {
        self.labels.get(label).copied()
    }

    /// Which `-i` input a node's frames descend from, so an output repeats the
    /// right stream header.
    pub fn root(&self, node: usize) -> usize {
        self.roots[node]
    }

    /// A node's name, for a message.
    pub fn name(&self, node: usize) -> &str {
        &self.nodes[node].name
    }

    /// Whether any module of this network acts on the rows arriving with its
    /// frames.
    pub fn any_module_reads_rows(&self) -> bool {
        self.nodes.iter().any(|node| node.reads_rows)
    }

    /// A module that reads rows the arriving annotations cannot reach, and the
    /// module that stops them: `(reader, stopped by)`. None when every reader
    /// can be reached.
    ///
    /// Rows arrive on an input, and travel on through a module only when it
    /// declares that upstream rows may leave on its output frames. Nodes are
    /// in topological order, so one pass settles which of them see the
    /// arriving rows.
    pub fn reader_the_rows_cannot_reach(&self) -> Option<(&str, &str)> {
        let mut sees = vec![false; self.nodes.len()];
        for (index, node) in self.nodes.iter().enumerate() {
            sees[index] = node.sources.iter().any(|source| match source {
                Source::Input(_) => true,
                Source::Node(upstream) => sees[*upstream] && self.nodes[*upstream].forwards_rows,
            });
        }
        let reader =
            (0..self.nodes.len()).find(|index| self.nodes[*index].reads_rows && !sees[*index])?;
        let blocker = self.stops_the_rows(reader, &sees)?;
        Some((
            self.nodes[reader].name.as_str(),
            self.nodes[blocker].name.as_str(),
        ))
    }

    /// The nearest module upstream of `reader` that sees the arriving rows and
    /// does not pass them on. A reader the rows cannot reach always has one.
    fn stops_the_rows(&self, reader: usize, sees: &[bool]) -> Option<usize> {
        let mut walked = vec![false; self.nodes.len()];
        let mut stack = vec![reader];
        while let Some(index) = stack.pop() {
            for source in &self.nodes[index].sources {
                let Source::Node(upstream) = source else {
                    continue;
                };
                if sees[*upstream] && !self.nodes[*upstream].forwards_rows {
                    return Some(*upstream);
                }
                if !walked[*upstream] {
                    walked[*upstream] = true;
                    stack.push(*upstream);
                }
            }
        }
        None
    }

    /// Every module in the network, named.
    pub fn modules(&self) -> String {
        self.nodes
            .iter()
            .map(|node| node.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Feeds one batch - `arriving[i]` the frames read off input i - through
    /// the whole network, returning what each node produced.
    pub fn advance(&mut self, arriving: Vec<Vec<Frame>>) -> Result<Vec<Vec<Frame>>> {
        let produced = vec![Vec::new(); self.nodes.len()];
        self.propagate(arriving, produced, 0)
    }

    /// Every node's final call in turn, whatever it produces flowing on
    /// through the nodes after it. `arriving_trailing[i]` is the rows that
    /// came in on input i with no frame to ride; they reach every module
    /// reading that input, on its final call, and one module's trailing rows
    /// become the final call's for every module reading it.
    pub fn drain(&mut self, arriving_trailing: &[Vec<String>]) -> Result<Drained> {
        for node in self.nodes.iter_mut() {
            for source in &node.sources {
                if let Source::Input(input) = source {
                    if let Some(rows) = arriving_trailing.get(*input) {
                        node.trailing.extend(rows.iter().cloned());
                    }
                }
            }
        }

        let mut frames: Vec<Vec<Frame>> = vec![Vec::new(); self.nodes.len()];
        let mut trailing: Vec<Vec<String>> = vec![Vec::new(); self.nodes.len()];
        for index in 0..self.nodes.len() {
            let finished = self.nodes[index].finish();
            let processed = finished
                .with_context(|| format!("{}: the final window", self.nodes[index].name))?;

            for later in index + 1..self.nodes.len() {
                if self.nodes[later].sources.contains(&Source::Node(index)) {
                    let rows = processed.trailing.clone();
                    self.nodes[later].trailing.extend(rows);
                }
            }
            trailing[index] = processed.trailing;

            let mut produced = vec![Vec::new(); self.nodes.len()];
            produced[index] = processed.frames;
            let pass = self.propagate(Vec::new(), produced, index + 1)?;
            for (accumulated, frames) in frames.iter_mut().zip(pass) {
                accumulated.extend(frames);
            }
        }
        Ok(Drained { frames, trailing })
    }

    /// One walk over the nodes from `start` on. `produced[k]` is what node k
    /// has already put out in this pass; a node before `start` has been
    /// drained already and hands on nothing.
    ///
    /// A stream's last reader takes its frames; every reader before it copies
    /// them, which is what handing one module's output to two readers costs.
    fn propagate(
        &mut self,
        mut arriving: Vec<Vec<Frame>>,
        mut produced: Vec<Vec<Frame>>,
        start: usize,
    ) -> Result<Vec<Vec<Frame>>> {
        for index in start..self.nodes.len() {
            let sources = self.nodes[index].sources.clone();
            let mut arrived: Vec<Vec<Frame>> = Vec::with_capacity(sources.len());
            for (pad, source) in sources.iter().enumerate() {
                // One source may feed several of this node's own pads - the
                // shape a module reading one stream twice takes - so only the
                // last pad reading it may take rather than copy.
                let last = self.last_reader.get(source) == Some(&index)
                    && !sources[pad + 1..].contains(source);
                arrived.push(match *source {
                    Source::Input(input) => match arriving.get_mut(input) {
                        Some(frames) if last => std::mem::take(frames),
                        Some(frames) => frames.clone(),
                        None => Vec::new(),
                    },
                    Source::Node(node) if last => std::mem::take(&mut produced[node]),
                    Source::Node(node) => produced[node].clone(),
                });
            }
            // No context here: everything `feed` can fail on already names the
            // module it failed in.
            let out = self.nodes[index].feed(arrived)?;
            produced[index].extend(out);
        }
        Ok(produced)
    }
}

/// The last node reading each stream, in topological order. A node whose
/// frames an output is mapped to is left out: the walk must not take those.
fn last_readers(nodes: &[Node], sunk: &[usize]) -> HashMap<Source, usize> {
    let mut last = HashMap::new();
    for (index, node) in nodes.iter().enumerate() {
        for source in &node.sources {
            if let Source::Node(upstream) = source {
                if sunk.contains(upstream) {
                    continue;
                }
            }
            last.insert(*source, index);
        }
    }
    last
}

/// The path each bound name loads, refusing a name bound twice.
fn bound_paths(bindings: &[Binding]) -> Result<HashMap<&str, String>> {
    let mut paths: HashMap<&str, String> = HashMap::new();
    for binding in bindings {
        if paths.contains_key(binding.name.as_str()) {
            bail!("-m binds the name '{}' twice", binding.name);
        }
        paths.insert(binding.name.as_str(), binding.path.clone());
    }
    Ok(paths)
}

fn bound_names(bindings: &[Binding]) -> String {
    if bindings.is_empty() {
        return "nothing".to_string();
    }
    bindings
        .iter()
        .map(|b| b.name.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// The node each label is written by, refusing a label written twice.
fn writers(parsed: &[ParsedNode]) -> Result<HashMap<&str, usize>> {
    let mut writer: HashMap<&str, usize> = HashMap::new();
    for (index, node) in parsed.iter().enumerate() {
        for label in &node.outputs {
            if let Some(previous) = writer.get(label.as_str()) {
                bail!(
                    "label [{label}] is written by both '{}' and '{}'",
                    parsed[*previous].module,
                    node.module
                );
            }
            writer.insert(label.as_str(), index);
        }
    }
    Ok(writer)
}

/// Every label a node writes must be read: by another node, or by a `-map`.
fn check_consumed(
    parsed: &[ParsedNode],
    mapped: &[String],
    writer: &HashMap<&str, usize>,
) -> Result<()> {
    let mut read: Vec<&str> = mapped.iter().map(String::as_str).collect();
    for node in parsed {
        for pad in &node.inputs {
            if let Pad::Label(label) = pad {
                read.push(label.as_str());
            }
        }
    }
    for (label, index) in writer {
        if !read.contains(label) {
            bail!(
                "label [{label}] is written by '{}' and nothing reads it",
                parsed[*index].module
            );
        }
    }
    Ok(())
}

/// Each node's inputs as sources, in the parsed node numbering.
fn resolve(
    parsed: &[ParsedNode],
    writer: &HashMap<&str, usize>,
    inputs: usize,
) -> Result<Vec<Vec<Source>>> {
    let mut resolved = Vec::with_capacity(parsed.len());
    for node in parsed {
        if node.inputs.is_empty() {
            bail!("module '{}' is given no input stream", node.module);
        }
        let mut sources = Vec::with_capacity(node.inputs.len());
        for pad in &node.inputs {
            sources.push(match pad {
                Pad::Input { index, .. } => {
                    if *index >= inputs {
                        bail!(
                            "module '{}' reads {} and {inputs} input(s) were given with -i",
                            node.module,
                            spell(pad)
                        );
                    }
                    Source::Input(*index)
                }
                Pad::Label(label) => {
                    let producer = writer.get(label.as_str()).ok_or_else(|| {
                        anyhow!(
                            "module '{}' reads label [{label}] and nothing writes it",
                            node.module
                        )
                    })?;
                    Source::Node(*producer)
                }
            });
        }
        resolved.push(sources);
    }
    Ok(resolved)
}

/// An input nothing reads would be opened and drained for nothing, which is a
/// wiring mistake rather than a shape this host serves.
fn check_inputs_read(sources: &[Vec<Source>], inputs: usize) -> Result<()> {
    for index in 0..inputs {
        if !sources
            .iter()
            .flatten()
            .any(|source| *source == Source::Input(index))
        {
            bail!("input {index} is given with -i and no module reads [{index}:v] or [{index}:a]");
        }
    }
    Ok(())
}

/// Node indices in dependency order, ties broken by the order the chains were
/// written - so a graph already in topological order comes back in its own.
fn topological(parsed: &[ParsedNode], sources: &[Vec<Source>]) -> Result<Vec<usize>> {
    let mut order = Vec::with_capacity(parsed.len());
    let mut placed = vec![false; parsed.len()];

    while order.len() < parsed.len() {
        let next = (0..parsed.len()).find(|index| {
            !placed[*index]
                && sources[*index].iter().all(|source| match source {
                    Source::Input(_) => true,
                    Source::Node(upstream) => placed[*upstream],
                })
        });
        let Some(next) = next else {
            let stuck: Vec<&str> = (0..parsed.len())
                .filter(|index| !placed[*index])
                .map(|index| parsed[index].module.as_str())
                .collect();
            bail!(
                "the network has a cycle through {}; frames would never reach an output",
                stuck.join(", ")
            );
        };
        placed[next] = true;
        order.push(next);
    }
    Ok(order)
}

/// The `-i` input a node's frames descend from: its first source's, since a
/// module never changes the geometry it was opened for.
fn root_input(sources: &[Source], roots: &[usize]) -> usize {
    match sources[0] {
        Source::Input(index) => index,
        Source::Node(index) => roots[index],
    }
}

/// Every pad a node reads must carry the same payloads, since one module is
/// opened for one geometry. That covers the kind as well: a video pad and an
/// audio one never match.
fn check_one_format(
    module: &str,
    sources: &[Source],
    roots: &[usize],
    formats: &[Format],
) -> Result<()> {
    let first = formats[root_input(sources, roots)];
    for (pad, source) in sources.iter().enumerate().skip(1) {
        let other = formats[root_input(std::slice::from_ref(source), roots)];
        if other.media != first.media {
            bail!(
                "module '{module}' reads {} on pad 0 and {} on pad {pad}; every pad of a module \
                 carries what it was opened for",
                describe_media(&first),
                describe_media(&other)
            );
        }
    }
    Ok(())
}

/// One stream's geometry, for a message.
fn describe_media(format: &Format) -> String {
    match format.media {
        Media::Video(video) => format!(
            "a {}x{} {} stream",
            video.width, video.height, video.pix_fmt
        ),
        Media::Audio(audio) => format!(
            "a {} channel {} stream at {} Hz",
            audio.channels, audio.sample_fmt, audio.sample_rate
        ),
    }
}

/// An input label says which kind it reads, and the stream on that `-i` says
/// which kind it carries. A mismatch is refused naming the edge.
fn check_input_kinds(parsed: &[ParsedNode], formats: &[Format]) -> Result<()> {
    for node in parsed {
        for pad in &node.inputs {
            let Pad::Input { index, kind } = pad else {
                continue;
            };
            let Some(format) = formats.get(*index) else {
                continue;
            };
            let carried = match format.kind() {
                Kind::Video => EdgeKind::Video,
                Kind::Audio => EdgeKind::Audio,
            };
            if carried != *kind {
                bail!(
                    "the network reads {} and input {index} carries {}",
                    spell(pad),
                    format.kind()
                );
            }
        }
    }
    Ok(())
}

/// The pads wired to a module against the streams it says it reads. A module
/// handed the wrong number of them would be driven out of lockstep, so the
/// mismatch is refused naming the module and both counts.
fn check_pad_count(module: &str, reads: usize, wired: usize) -> Result<()> {
    if wired == reads {
        return Ok(());
    }
    bail!(
        "module '{module}' is wired {wired} input(s) and reads {reads}; a module is given one \
         stream per pad it declares"
    );
}

/// A module reads the kind it publishes. An audio edge into a video module,
/// or the other way about, is refused naming the edge and the module.
fn check_module_kind(node: &ParsedNode, described: &Described, carried: Kind) -> Result<()> {
    let declared = described.meta.kind()?;
    if declared == carried {
        return Ok(());
    }
    let edge = node
        .inputs
        .first()
        .map(spell)
        .unwrap_or_else(|| "its input".to_string());
    bail!(
        "{edge} carries {carried} and module '{}' is {} module; an edge feeds a module of its own \
         kind",
        node.module,
        article(declared)
    );
}

/// `a` or `an` before a kind, and the kind.
fn article(kind: Kind) -> String {
    match kind {
        Kind::Video => "a video".to_string(),
        Kind::Audio => "an audio".to_string(),
    }
}

/// A pad as it is written in the wiring.
fn spell(pad: &Pad) -> String {
    match pad {
        Pad::Input { index, kind } => format!("[{index}:{}]", kind.marker()),
        Pad::Label(label) => format!("[{label}]"),
    }
}

/// A module's `params-schema` as JSON. An empty schema is an object with no
/// properties, which is a module that takes nothing.
fn parse_schema(schema: &str, module: &str) -> Result<serde_json::Value> {
    if schema.trim().is_empty() {
        return Ok(serde_json::Value::Object(serde_json::Map::new()));
    }
    serde_json::from_str(schema)
        .with_context(|| format!("module '{module}': its params schema is not valid JSON"))
}

/// The options a network node was written with, as the JSON object the module
/// takes: each value read as the type its schema declares. Returns an empty
/// string for a node given no options, which is what a module with no `-params`
/// sees.
fn params_json(
    module: &str,
    schema: &serde_json::Value,
    options: &[(String, String)],
) -> Result<String> {
    if options.is_empty() {
        return Ok(String::new());
    }
    let properties = schema.get("properties").and_then(|p| p.as_object());
    let mut params = serde_json::Map::new();

    for (key, value) in options {
        if params.contains_key(key) {
            bail!("module '{module}' is given the option '{key}' twice");
        }
        let property = properties.and_then(|p| p.get(key)).ok_or_else(|| {
            anyhow!(
                "module '{module}' has no parameter '{key}'; it declares {}",
                declared(properties)
            )
        })?;
        params.insert(key.clone(), read_value(module, key, value, property)?);
    }
    serde_json::to_string(&serde_json::Value::Object(params))
        .with_context(|| format!("module '{module}': writing its parameters as JSON"))
}

/// The parameter names a schema declares, for a refusal.
fn declared(properties: Option<&serde_json::Map<String, serde_json::Value>>) -> String {
    match properties {
        Some(properties) if !properties.is_empty() => properties
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join(", "),
        _ => "none".to_string(),
    }
}

/// One option's text as the JSON type its schema declares. A schema this host
/// does not judge takes the text as written.
fn read_value(
    module: &str,
    key: &str,
    value: &str,
    property: &serde_json::Value,
) -> Result<serde_json::Value> {
    let wanted = property.get("type").and_then(|t| t.as_str());
    match wanted {
        Some("integer") => value
            .parse::<i64>()
            .map(serde_json::Value::from)
            .map_err(|_| bad_value(module, key, value, "an integer")),
        Some("number") => value
            .parse::<f64>()
            .ok()
            .and_then(serde_json::Number::from_f64)
            .map(serde_json::Value::Number)
            .ok_or_else(|| bad_value(module, key, value, "a number")),
        Some("boolean") => match value {
            "1" | "true" => Ok(serde_json::Value::Bool(true)),
            "0" | "false" => Ok(serde_json::Value::Bool(false)),
            _ => Err(bad_value(module, key, value, "true or false")),
        },
        _ => Ok(serde_json::Value::String(value.to_string())),
    }
}

fn bad_value(module: &str, key: &str, value: &str, wanted: &str) -> anyhow::Error {
    anyhow!("module '{module}': option '{key}={value}' is not {wanted}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "radius": {"type": "integer"},
                "threshold": {"type": "number"},
                "sharp": {"type": "boolean"},
                "label": {"type": "string"},
            }
        })
    }

    fn option(key: &str, value: &str) -> Vec<(String, String)> {
        vec![(key.to_string(), value.to_string())]
    }

    #[test]
    fn a_node_with_no_options_is_given_no_params() {
        assert_eq!(params_json("blur", &schema(), &[]).expect("no options"), "");
    }

    #[test]
    fn each_option_takes_the_type_its_schema_declares() {
        let options = vec![
            ("radius".to_string(), "8".to_string()),
            ("threshold".to_string(), "1.5".to_string()),
            ("sharp".to_string(), "1".to_string()),
            ("label".to_string(), "12".to_string()),
        ];
        let json = params_json("blur", &schema(), &options).expect("every option is declared");
        assert_eq!(
            json,
            r#"{"label":"12","radius":8,"sharp":true,"threshold":1.5}"#
        );
    }

    #[test]
    fn an_undeclared_option_names_what_the_module_takes() {
        let err = params_json("blur", &schema(), &option("nope", "1")).expect_err("refused");
        let message = err.to_string();
        assert!(message.contains("no parameter 'nope'"), "got: {message}");
        assert!(message.contains("radius"), "got: {message}");
    }

    #[test]
    fn a_value_of_the_wrong_type_is_refused_by_name() {
        let err = params_json("blur", &schema(), &option("radius", "wide")).expect_err("refused");
        let message = err.to_string();
        assert!(message.contains("radius=wide"), "got: {message}");
        assert!(message.contains("an integer"), "got: {message}");
    }
}
