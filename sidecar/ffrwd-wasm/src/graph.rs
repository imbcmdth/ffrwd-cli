//! The `-filter_complex` string a module network is wired by.
//!
//! The grammar is the subset of ffmpeg's own that the compiler emits: chains
//! separated by `;`, one module per chain, each chain a run of input labels,
//! the module with its options, then its output labels. An input stream is
//! `[N:v]` or `[N:a]`, where N indexes the `-i` list and the marker says which
//! kind it carries; anything else in brackets is a label another chain
//! writes.
//!
//! Values are unescaped the way ffmpeg unescapes them: once when the graph is
//! split into filters (`\` escapes, `'` quotes, `[ ] , ;` separate), then again
//! when a filter's option string is split into `k=v` pairs (`\` escapes, `'`
//! quotes, `:` separates). Both levels are undone here, in that order.

use anyhow::{anyhow, bail, Result};

/// Which kind an input label says it reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    Video,
    Audio,
}

impl EdgeKind {
    /// The letter the label carries: `[0:v]` or `[0:a]`.
    pub fn marker(&self) -> &'static str {
        match self {
            EdgeKind::Video => VIDEO_MARKER,
            EdgeKind::Audio => AUDIO_MARKER,
        }
    }
}

/// One pad a node reads: a `-i` input, or a label another node writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pad {
    Input { index: usize, kind: EdgeKind },
    Label(String),
}

/// One module of a network: what it is, how it is configured, and what it
/// reads and writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedNode {
    /// The name an `-m name=path` binding gives the module.
    pub module: String,
    /// Its options, in the order they were written.
    pub options: Vec<(String, String)>,
    pub inputs: Vec<Pad>,
    pub outputs: Vec<String>,
}

/// What a bracketed label turned out to name.
enum Token {
    Label(String),
    Filter(String),
    ChainEnd,
    Merge,
}

/// The stream type markers a network input label carries.
const VIDEO_MARKER: &str = "v";
const AUDIO_MARKER: &str = "a";

/// Reads a whole `-filter_complex` string as a list of nodes, in the order
/// the chains were written.
pub fn parse_network(text: &str) -> Result<Vec<ParsedNode>> {
    if text.trim().is_empty() {
        bail!("-filter_complex is empty; a module network names at least one module");
    }
    let tokens = tokenize(text)?;

    let mut nodes = Vec::new();
    let mut chain: Vec<Token> = Vec::new();
    for token in tokens {
        match token {
            Token::ChainEnd => {
                nodes.push(chain_node(&chain, nodes.len())?);
                chain = Vec::new();
            }
            other => chain.push(other),
        }
    }
    nodes.push(chain_node(&chain, nodes.len())?);
    Ok(nodes)
}

/// One `;`-separated chain as a node. `index` names it in a refusal.
fn chain_node(chain: &[Token], index: usize) -> Result<ParsedNode> {
    if chain.iter().any(|t| matches!(t, Token::Merge)) {
        bail!(
            "chain {index} of -filter_complex merges filters with ','; \
             a module network spells one module per chain, separated by ';'"
        );
    }

    let mut inputs = Vec::new();
    let mut filter: Option<&String> = None;
    let mut outputs = Vec::new();
    for token in chain {
        match token {
            Token::Label(label) => {
                if filter.is_none() {
                    inputs.push(input_pad(label, index)?);
                } else {
                    outputs.push(label.clone());
                }
            }
            Token::Filter(text) => {
                if filter.is_some() {
                    bail!("chain {index} of -filter_complex names two modules in a row");
                }
                filter = Some(text);
            }
            Token::Merge | Token::ChainEnd => {
                unreachable!("chains are cut on ';', and a merged one was refused above")
            }
        }
    }

    let Some(text) = filter else {
        bail!("chain {index} of -filter_complex names no module");
    };
    let (module, options) = split_filter(text, index)?;
    Ok(ParsedNode {
        module,
        options,
        inputs,
        outputs,
    })
}

/// A chain's leading label: an input stream, or another chain's output.
fn input_pad(label: &str, index: usize) -> Result<Pad> {
    let Some((number, marker)) = label.split_once(':') else {
        return Ok(Pad::Label(label.to_string()));
    };
    if marker.contains(':') {
        bail!(
            "chain {index} of -filter_complex reads [{label}], which carries a \
             per-type index; a network input carries one stream, so it is [{number}:{VIDEO_MARKER}]"
        );
    }
    let kind = match marker {
        VIDEO_MARKER => EdgeKind::Video,
        AUDIO_MARKER => EdgeKind::Audio,
        _ => bail!(
            "chain {index} of -filter_complex reads [{label}]; a module network hosts video and \
             audio, so an input label is [{number}:{VIDEO_MARKER}] or [{number}:{AUDIO_MARKER}]"
        ),
    };
    let input: usize = number.parse().map_err(|_| {
        anyhow!(
            "chain {index} of -filter_complex reads [{label}], whose input number is not a number"
        )
    })?;
    Ok(Pad::Input { index: input, kind })
}

/// `name=k=v:k2=v2` split into the module name and its options. A module name
/// is spelled the way a `-m` binding spells it, so the first `=` separates.
fn split_filter(text: &str, index: usize) -> Result<(String, Vec<(String, String)>)> {
    let (name, rest) = match text.split_once('=') {
        Some((name, rest)) => (name, Some(rest)),
        None => (text, None),
    };
    if name.is_empty() {
        bail!("chain {index} of -filter_complex names an empty module");
    }
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        bail!(
            "chain {index} of -filter_complex names module '{name}'; \
             a bound module name holds letters, digits and underscores only"
        );
    }
    let options = match rest {
        Some(rest) => split_options(rest, name)?,
        None => Vec::new(),
    };
    Ok((name.to_string(), options))
}

/// One filter's option string as `k=v` pairs, with the option level's quoting
/// and escaping undone.
fn split_options(text: &str, module: &str) -> Result<Vec<(String, String)>> {
    let mut options = Vec::new();
    for part in split_escaped(text, ':') {
        if part.raw.is_empty() {
            continue;
        }
        let Some(at) = part.separator else {
            bail!(
                "module '{module}' is given the option '{}', which is not k=v",
                unescape(&part.raw)
            );
        };
        let key = unescape(&part.raw[..at]);
        let value = unescape(&part.raw[at + 1..]);
        if key.is_empty() {
            bail!("module '{module}' is given an option with an empty name");
        }
        options.push((key, value));
    }
    Ok(options)
}

/// One piece of a split, still escaped, and where its first unescaped `=` is.
struct Part {
    raw: String,
    separator: Option<usize>,
}

/// Splits `text` on `sep`, honouring `\` escapes and `'` quotes, and notes each
/// piece's first `=` that is neither escaped nor quoted.
fn split_escaped(text: &str, sep: char) -> Vec<Part> {
    let mut parts = Vec::new();
    let mut raw = String::new();
    let mut separator: Option<usize> = None;
    let mut quoted = false;
    let mut chars = text.chars();

    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                raw.push(c);
                if let Some(next) = chars.next() {
                    raw.push(next);
                }
            }
            '\'' => {
                quoted = !quoted;
                raw.push(c);
            }
            '=' if !quoted && separator.is_none() => {
                separator = Some(raw.len());
                raw.push(c);
            }
            c if c == sep && !quoted => {
                parts.push(Part { raw, separator });
                raw = String::new();
                separator = None;
            }
            c => raw.push(c),
        }
    }
    parts.push(Part { raw, separator });
    parts
}

/// One level of ffmpeg escaping removed: `\X` becomes X and `'...'` becomes
/// its contents.
fn unescape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                if let Some(next) = chars.next() {
                    out.push(next);
                }
            }
            '\'' => {}
            c => out.push(c),
        }
    }
    out
}

/// Splits the graph string into labels, filters and separators, removing the
/// graph level's escaping as it goes.
fn tokenize(text: &str) -> Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            c if c.is_whitespace() => i += 1,
            '[' => {
                let start = i + 1;
                let end = chars[start..]
                    .iter()
                    .position(|c| *c == ']')
                    .map(|offset| start + offset);
                let Some(end) = end else {
                    bail!("-filter_complex has a '[' that is never closed");
                };
                let label: String = chars[start..end].iter().collect();
                if label.is_empty() {
                    bail!("-filter_complex has an empty label");
                }
                tokens.push(Token::Label(label));
                i = end + 1;
            }
            ']' => bail!("-filter_complex has a ']' that opens nothing"),
            ';' => {
                tokens.push(Token::ChainEnd);
                i += 1;
            }
            ',' => {
                tokens.push(Token::Merge);
                i += 1;
            }
            _ => {
                let (text, next) = read_filter(&chars, i);
                tokens.push(Token::Filter(text));
                i = next;
            }
        }
    }
    Ok(tokens)
}

/// A filter's text, up to the next structural character, with the graph
/// level's escaping removed. Returns where scanning stopped.
fn read_filter(chars: &[char], from: usize) -> (String, usize) {
    let mut out = String::new();
    let mut quoted = false;
    let mut i = from;
    // Whitespace this level did not escape is layout, not value, so a run of
    // it before the next label is dropped.
    let mut trailing = 0usize;

    while i < chars.len() {
        let c = chars[i];
        match c {
            '\\' => {
                if let Some(next) = chars.get(i + 1) {
                    out.push(*next);
                    trailing = 0;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            '\'' => {
                quoted = !quoted;
                i += 1;
            }
            '[' | ']' | ',' | ';' if !quoted => break,
            c => {
                out.push(c);
                trailing = if c.is_whitespace() {
                    trailing + c.len_utf8()
                } else {
                    0
                };
                i += 1;
            }
        }
    }
    out.truncate(out.len() - trailing);
    (out, i)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    #[test]
    fn the_faces_network_reads_as_two_nodes() {
        let nodes = parse_network("[0:v]facebox[n1];[n1]blur_boxes[out0]").expect("parses");
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].module, "facebox");
        assert_eq!(
            nodes[0].inputs,
            vec![Pad::Input {
                index: 0,
                kind: EdgeKind::Video
            }]
        );
        assert_eq!(nodes[0].outputs, labels(&["n1"]));
        assert_eq!(nodes[1].module, "blur_boxes");
        assert_eq!(nodes[1].inputs, vec![Pad::Label("n1".to_string())]);
        assert_eq!(nodes[1].outputs, labels(&["out0"]));
    }

    #[test]
    fn one_label_may_be_read_by_two_chains() {
        let nodes = parse_network("[0:v]invert[a];[a]double[out0];[a]tail3[out1]").expect("parses");
        assert_eq!(nodes.len(), 3);
        assert_eq!(nodes[1].inputs, vec![Pad::Label("a".to_string())]);
        assert_eq!(nodes[2].inputs, vec![Pad::Label("a".to_string())]);
    }

    #[test]
    fn options_come_back_as_pairs() {
        let nodes = parse_network("[0:v]facebox=min-face-size=20:score-threshold=1.5[out0]")
            .expect("parses");
        assert_eq!(
            nodes[0].options,
            vec![
                ("min-face-size".to_string(), "20".to_string()),
                ("score-threshold".to_string(), "1.5".to_string()),
            ]
        );
    }

    #[test]
    fn both_levels_of_escaping_are_undone() {
        // What the compiler writes for the value `a:b'c` - escaped once for
        // the option list, then again for the graph.
        let nodes = parse_network(r"[0:v]brand=title=a\\:b\\\'c[out0]").expect("parses");
        assert_eq!(
            nodes[0].options,
            vec![("title".to_string(), "a:b'c".to_string())]
        );
    }

    /// One value escaped the way the compiler escapes it: once for the option
    /// list, then again for the graph.
    fn escape(value: &str) -> String {
        let mut once = String::new();
        for c in value.chars() {
            if matches!(c, '\\' | '\'' | ':') {
                once.push('\\');
            }
            once.push(c);
        }
        let mut twice = String::new();
        for c in once.chars() {
            if matches!(c, '\\' | '\'' | '[' | ']' | ',' | ';') {
                twice.push('\\');
            }
            twice.push(c);
        }
        twice
    }

    #[test]
    fn a_rows_predicate_survives_the_network_string_round_trip() {
        // Quotes, colons, commas and braces, which is every character the two
        // levels of escaping have an opinion about.
        let pred = r#"{"and":[{"eq":[{"field":"class"},{"lit":"person"}]},{"ge":[{"field":"score"},{"lit":0.25}]}]}"#;
        let text = format!("[a]rowfilter=pred={}[b]", escape(pred));
        let nodes = parse_network(&text).expect("parses");
        assert_eq!(nodes[0].module, "rowfilter");
        assert_eq!(
            nodes[0].options,
            vec![("pred".to_string(), pred.to_string())],
            "the predicate arrives exactly as it was written"
        );
    }

    #[test]
    fn a_predicate_carrying_a_quoted_colon_survives_as_written() {
        // A string value with the characters both levels split on inside it.
        let pred = r#"{"eq":[{"field":"label"},{"lit":"a:b'c,d"}]}"#;
        let text = format!("[a]rowfilter=pred={}[b]", escape(pred));
        let nodes = parse_network(&text).expect("parses");
        assert_eq!(nodes[0].options[0].1, pred);
    }

    #[test]
    fn an_empty_value_survives_its_quotes() {
        let nodes = parse_network("[0:v]brand=title=''[out0]").expect("parses");
        assert_eq!(nodes[0].options, vec![("title".to_string(), String::new())]);
    }

    #[test]
    fn a_comma_merged_chain_is_refused() {
        let err = parse_network("[0:v]invert,double[out0]").expect_err("refused");
        let message = err.to_string();
        assert!(message.contains("','"), "got: {message}");
        assert!(message.contains("one module per chain"), "got: {message}");
    }

    #[test]
    fn a_per_type_input_index_is_refused_by_name() {
        let err = parse_network("[0:v:0]invert[out0]").expect_err("refused");
        assert!(err.to_string().contains("[0:v]"), "got: {err}");
    }

    #[test]
    fn an_audio_input_reads_as_an_audio_edge() {
        let nodes = parse_network("[0:a]again[out0]").expect("parses");
        assert_eq!(
            nodes[0].inputs,
            vec![Pad::Input {
                index: 0,
                kind: EdgeKind::Audio
            }],
            "the marker is what says which kind the edge carries"
        );
    }

    #[test]
    fn a_marker_that_is_neither_kind_is_refused_naming_both() {
        let err = parse_network("[0:s]invert[out0]").expect_err("refused");
        let message = err.to_string();
        assert!(message.contains("[0:s]"), "got: {message}");
        assert!(
            message.contains("[0:v]") && message.contains("[0:a]"),
            "got: {message}"
        );
    }

    #[test]
    fn a_chain_with_no_module_is_refused() {
        let err = parse_network("[0:v][out0]").expect_err("refused");
        assert!(err.to_string().contains("names no module"), "got: {err}");
    }

    #[test]
    fn an_option_that_is_not_a_pair_is_refused() {
        let err = parse_network("[0:v]invert=lonely[out0]").expect_err("refused");
        let message = err.to_string();
        assert!(
            message.contains("lonely") && message.contains("k=v"),
            "got: {message}"
        );
    }
}
