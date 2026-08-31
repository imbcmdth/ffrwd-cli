//! A network of modules in one process: the DAG the `-filter_complex` string
//! describes, checked and opened.
//!
//! Opening settles everything static - every module name bound, every label
//! written once and read, kinds and pad counts matched, every module
//! instantiated - and hands the scheduler one lane seed per node in
//! topological order. The scheduler is what drives them.
//!
//! A chain of one module is a network of one, which is how the single-module
//! spelling stays the same code.

use std::collections::HashMap;

use anyhow::{anyhow, bail, Context, Result};
use ffrwd_wasm_runtime::runtime::{self, Described, Filter, Format, Kind, Media, StreamInfo};

use crate::graph::{EdgeKind, Pad, ParsedNode};
use crate::rowfilter::{self, RowFilter};
use crate::scheduler::{LaneSeed, Reopen, Runner};

/// Where one of a node's streams comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    /// A `-i` input, by index.
    Input(usize),
    /// Another node's output, by index in topological order.
    Node(usize),
}

/// One module bound to a name by `-m name=path`.
#[derive(Debug, Clone)]
pub struct Binding {
    pub name: String,
    pub path: String,
}

/// The opened network: one lane seed per node in topological order, the label
/// each one writes, and which input each descends from.
pub struct Network {
    seeds: Vec<LaneSeed>,
    /// Whether each node acts on the rows arriving with its frames.
    reads_rows: Vec<bool>,
    /// Whether upstream rows may leave on each node's own output frames.
    forwards_rows: Vec<bool>,
    labels: HashMap<String, usize>,
    roots: Vec<usize>,
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
        let mut seeds = Vec::with_capacity(order.len());
        let mut reads_rows = Vec::with_capacity(order.len());
        let mut forwards_rows = Vec::with_capacity(order.len());
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
            let format = formats[root];

            // The rows node is the host's own: nothing is compiled, and it
            // carries whichever kind reaches it, so neither the module kind
            // nor a params schema applies.
            let seed = if node.module == rowfilter::NODE {
                check_pad_count(&node.module, 1, placed.len())?;
                reads_rows.push(true);
                forwards_rows.push(true);
                LaneSeed {
                    name: rowfilter::NODE.to_string(),
                    runners: vec![Runner::Rows(RowFilter::open(&node.options)?)],
                    shape: rowfilter::SHAPE,
                    sources: placed,
                    format,
                    reopen: None,
                }
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
                check_module_kind(node, &described, format.kind())?;
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
                let filter = Filter::open(path, &format, &streams[root], &params)
                    .with_context(|| format!("opening module '{}' from {path}", node.module))?;
                reads_rows.push(filter.reads_rows());
                forwards_rows.push(filter.forwards_rows());
                let reopen = reopen_for(&filter, path, &params, &format, &streams[root]);
                LaneSeed {
                    name: filter.name().to_string(),
                    shape: filter.shape(),
                    runners: vec![Runner::Module(Box::new(filter))],
                    sources: placed,
                    format,
                    reopen,
                }
            };

            for label in &node.outputs {
                labels.insert(label.clone(), seeds.len());
            }
            roots.push(root);
            seeds.push(seed);
        }

        Ok(Network {
            seeds,
            reads_rows,
            forwards_rows,
            labels,
            roots,
        })
    }

    /// A network of one module, wired from one input to one node - the shape
    /// the single-module spelling runs as.
    pub fn single(filter: Filter, format: &Format, reopen: Option<Reopen>) -> Network {
        let reads = filter.reads_rows();
        let forwards = filter.forwards_rows();
        Network {
            seeds: vec![LaneSeed {
                name: filter.name().to_string(),
                shape: filter.shape(),
                runners: vec![Runner::Module(Box::new(filter))],
                sources: vec![Source::Input(0)],
                format: *format,
                reopen,
            }],
            reads_rows: vec![reads],
            forwards_rows: vec![forwards],
            labels: HashMap::new(),
            roots: vec![0],
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

    /// The lanes this network opened, in topological order, for the scheduler.
    pub fn into_seeds(self) -> Vec<LaneSeed> {
        self.seeds
    }

    /// Whether any module of this network acts on the rows arriving with its
    /// frames.
    pub fn any_module_reads_rows(&self) -> bool {
        self.reads_rows.iter().any(|reads| *reads)
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
        let mut sees = vec![false; self.seeds.len()];
        for (index, seed) in self.seeds.iter().enumerate() {
            sees[index] = seed.sources.iter().any(|source| match source {
                Source::Input(_) => true,
                Source::Node(upstream) => sees[*upstream] && self.forwards_rows[*upstream],
            });
        }
        let reader =
            (0..self.seeds.len()).find(|index| self.reads_rows[*index] && !sees[*index])?;
        let blocker = self.stops_the_rows(reader, &sees)?;
        Some((
            self.seeds[reader].name.as_str(),
            self.seeds[blocker].name.as_str(),
        ))
    }

    /// The nearest module upstream of `reader` that sees the arriving rows and
    /// does not pass them on. A reader the rows cannot reach always has one.
    fn stops_the_rows(&self, reader: usize, sees: &[bool]) -> Option<usize> {
        let mut walked = vec![false; self.seeds.len()];
        let mut stack = vec![reader];
        while let Some(index) = stack.pop() {
            for source in &self.seeds[index].sources {
                let Source::Node(upstream) = source else {
                    continue;
                };
                if sees[*upstream] && !self.forwards_rows[*upstream] {
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
        self.seeds
            .iter()
            .map(|seed| seed.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// How a pure video module's lane grows more instances: reopened from the
/// same path with the same parameters. Everything else keeps its one.
pub fn reopen_for(
    filter: &Filter,
    path: &str,
    params: &str,
    format: &Format,
    info: &StreamInfo,
) -> Option<Reopen> {
    if !filter.shape().pure || format.video().is_none() {
        return None;
    }
    Some(Reopen {
        path: path.to_string(),
        params: params.to_string(),
        format: *format,
        info: info.clone(),
    })
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
