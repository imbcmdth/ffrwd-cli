//! `ffrwd-wasm relay`: the process that moves an edge's bytes between two
//! others, so the host that names the pipes never copies a byte of them.
//!
//! One relay serves a whole run on one machine. It takes no edges on argv:
//! the host writes one JSON object per line on its stdin, and the relay
//! answers with rows on stderr, each a line behind [`ROW_PREFIX`]:
//!
//! ```text
//! {"edges": [{"id": ID, "from": END, "to": END, "depth": N, "buffer": N, "spool": B}], "batch": N}
//!     makes every pipe of the batch, starts its edges, and answers
//!     {"kind":"relay","ready":N}, or {"kind":"relay","batch":N,"error":"..."}
//!     for a batch it could not set up, after which it carries on
//! {"stop": [ID, ...]}
//!     aborts those edges: both ends close at once, whatever is held
//! {"listen": {"host": HOST, "keys": [KEY, ...]}}
//!     opens the node's data port for the cut edges other nodes send here,
//!     and answers {"kind":"relay","listening":[HOST, PORT]}
//! stdin closed
//!     aborts every edge still open, unlinks every FIFO, and exits 0
//! ```
//!
//! An END is a pipe PATH the relay makes and serves, or one side of an edge
//! cut between nodes. As `from`, a PATH is a pipe the relay READS (a producer
//! opens it and writes) and `{"listen": KEY}` is the connection another node
//! dials in for KEY. As `to`, a PATH is a pipe the relay WRITES (a consumer
//! opens it and reads) and `{"dial": [HOST, PORT], "key": KEY}` is a
//! connection to the node that listens for KEY. `buffer` sizes an edge's
//! pipes (64 KiB when it is not given), `depth` bounds how far the relay
//! reads ahead of the consumer (64 KiB likewise), and a `spool` edge, a rows
//! document, is never bounded at all.
//!
//! Each edge is a reader thread and a writer thread with a queue between
//! them. The reader starts the moment the producer opens its end, before
//! the consumer has opened its own: a consumer may still be opening an
//! earlier input, and a producer nobody reads fills its pipe and stops
//! before it writes the output that earlier input is waiting for. A read
//! hands back whatever has arrived and the writer passes exactly that on;
//! waiting for a full buffer would hold back the tail of a frame the far
//! end of a fan is waiting to complete.
//!
//! What the host learns from an edge is a row whenever it changes:
//! `{"kind":"flow","edge":ID,"moved":N,"began":B,"writing":B,"opening":B,"done":B}`.
//! `moved` counts bytes handed on (on arrival for a spool), `began` turns on
//! with the first of them, `opening` is the producer's end open and the
//! consumer's not yet, and `done` is the edge's last row. `began`, `opening`
//! and `done` are reported the moment they change. `moved` and `writing` are
//! read four times a second, so a moving edge costs four rows a second and
//! not two per write; `writing` is whether a write to the consumer is
//! waiting when it is read, which is all the host's wedge detectors ask.
//!
//! An end that cannot be opened ends its edge, with a row saying why:
//! `{"kind":"relay","edge":ID,"error":"..."}`. A fault in the relay as a
//! whole is `{"kind":"relay","error":"..."}` and a nonzero exit.
//!
//! A cut edge's connection opens with one line from the dialing node,
//! `FFRWD-CUT 1 <secret> <key>`, and the listening node's `OK`. The secret is
//! the job's, from [`SECRET_ENV`], never argv. A connection the listener
//! will not take is closed, and reported as
//! `{"kind":"relay","refused":"<reason>"}`.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, Read, Seek, SeekFrom, Write};
use std::net::{Ipv6Addr, Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::leaky::ROW_PREFIX;

/// The most one read takes off a pipe, and the size of every buffer the
/// queue holds. A ceiling, not a quantum: a read returns what has arrived.
const CHUNK: usize = 1 << 16;

/// How far ahead of the consumer an edge reads when the host names no depth.
const DEFAULT_DEPTH: usize = 1 << 16;

/// Each pipe's own buffer, per direction, when the host names none: what
/// `pipes.DEFAULT_BUFFER` gives a pipe today.
const DEFAULT_BUFFER: u32 = 1 << 16;

/// How much of a spooled rows document is held in memory before the rest of
/// it goes to a temporary file.
const SPOOL_MEMORY: usize = 4 << 20;

/// How often a moving edge reports, and how often an abort that raced a
/// blocking call is repeated.
const REPORT_EVERY: Duration = Duration::from_millis(250);

/// How long a closed stdin waits for the aborted edges to let go of their
/// ends before the relay exits regardless.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(2);

/// The environment variable the job's secret arrives in.
const SECRET_ENV: &str = "FFRWD_NODE_SECRET";

/// What a cut edge's opening line starts with: the protocol and its version.
const CUT_PROTOCOL: &str = "FFRWD-CUT 1";

/// The longest opening line a listener reads, its newline included.
const OPENING_MOST: usize = 256;

/// How long a listener gives a connection to send its opening line.
const OPENING_WAIT: Duration = Duration::from_secs(5);

/// How long one attempt to reach a listening node waits, and how long a
/// refused one waits before the next.
const DIAL_WAIT: Duration = Duration::from_secs(1);
const DIAL_AGAIN: Duration = Duration::from_millis(50);

/// How often an edge waiting for its cut connection looks for an abort.
const ARRIVAL_POLL: Duration = Duration::from_millis(100);

/// Where an edge's bytes come from.
#[derive(Debug, Clone, PartialEq, Eq)]
enum FromEnd {
    /// A pipe the relay makes and reads.
    Pipe(String),
    /// The connection another node dials in for this key.
    Listen(String),
}

/// Where an edge's bytes go.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ToEnd {
    /// A pipe the relay makes and writes.
    Pipe(String),
    /// A connection to the node that listens for this key.
    Dial {
        host: String,
        port: u16,
        key: String,
    },
}

/// An END as a batch writes it: a path, or an object naming a cut.
#[derive(Deserialize)]
#[serde(untagged)]
enum EndLine {
    Path(String),
    Listen { listen: String },
    Dial { dial: (String, u16), key: String },
}

impl EndLine {
    fn pipe(path: String) -> Result<String, String> {
        if path.is_empty() {
            return Err("a pipe path that is empty".to_string());
        }
        Ok(path)
    }

    fn into_from(self) -> Result<FromEnd, String> {
        match self {
            EndLine::Path(path) => Ok(FromEnd::Pipe(EndLine::pipe(path)?)),
            EndLine::Listen { listen } => Ok(FromEnd::Listen(listen)),
            EndLine::Dial { .. } => {
                Err("a dial is where bytes go, so it is a \"to\" and not a \"from\"".to_string())
            }
        }
    }

    fn into_to(self) -> Result<ToEnd, String> {
        match self {
            EndLine::Path(path) => Ok(ToEnd::Pipe(EndLine::pipe(path)?)),
            EndLine::Dial {
                dial: (host, port),
                key,
            } => Ok(ToEnd::Dial { host, port, key }),
            EndLine::Listen { .. } => Err(
                "a listen is where bytes come from, so it is a \"from\" and not a \"to\""
                    .to_string(),
            ),
        }
    }
}

/// One edge as a batch writes it.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EdgeLine {
    id: String,
    from: Value,
    to: Value,
    #[serde(default)]
    depth: Option<u64>,
    #[serde(default)]
    buffer: Option<u64>,
    #[serde(default)]
    spool: bool,
}

/// One edge, checked.
struct Spec {
    id: String,
    from: FromEnd,
    to: ToEnd,
    depth: usize,
    buffer: u32,
    spool: bool,
}

/// One line of stdin.
enum Command {
    /// A batch's number, and its edges or what is wrong with them.
    Batch(u64, Result<Vec<Spec>, String>),
    Stop(Vec<String>),
    /// Where to open the data port and the keys it takes, or what is wrong
    /// with the command.
    Listen(Result<(String, Vec<String>), String>),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ListenLine {
    host: String,
    keys: Vec<String>,
}

/// What a line says. An error is a line the relay cannot answer at all:
/// with no batch number to reply under, the host would wait on a reply that
/// never comes, so that is a fault and not a batch error.
fn parse_command(line: &str) -> Result<Command, String> {
    let value: Value =
        serde_json::from_str(line).map_err(|e| format!("a command that is not JSON: {e}"))?;
    let Value::Object(mut fields) = value else {
        return Err("a command that is not a JSON object".to_string());
    };
    for only in ["stop", "listen"] {
        if fields.contains_key(only) && fields.len() > 1 {
            return Err(format!(
                "a \"{only}\" command takes nothing else, and this one has {}",
                quoted(fields.keys().filter(|k| *k != only))
            ));
        }
    }
    if let Some(ids) = fields.remove("stop") {
        let ids: Vec<String> = serde_json::from_value(ids)
            .map_err(|e| format!("\"stop\" is a list of edge ids: {e}"))?;
        return Ok(Command::Stop(ids));
    }
    if let Some(listen) = fields.remove("listen") {
        let listen = serde_json::from_value::<ListenLine>(listen)
            .map(|line| (line.host, line.keys))
            .map_err(|e| format!("\"listen\" is {{\"host\": HOST, \"keys\": [KEY, ...]}}: {e}"));
        return Ok(Command::Listen(listen));
    }
    let Some(batch) = fields.remove("batch") else {
        return Err(
            "a command that is none of a batch (\"batch\" and \"edges\"), a \"stop\" or a \"listen\""
                .to_string(),
        );
    };
    let batch = batch
        .as_u64()
        .ok_or_else(|| format!("\"batch\" is a whole number, not {batch}"))?;
    let edges = match fields.remove("edges") {
        _ if !fields.is_empty() => Err(format!(
            "a batch takes \"batch\" and \"edges\", not {}",
            quoted(fields.keys())
        )),
        None => Err("the batch has no \"edges\"".to_string()),
        Some(edges) => parse_edges(edges),
    };
    Ok(Command::Batch(batch, edges))
}

fn quoted<'a>(keys: impl Iterator<Item = &'a String>) -> String {
    keys.map(|k| format!("\"{k}\""))
        .collect::<Vec<_>>()
        .join(", ")
}

fn parse_edges(edges: Value) -> Result<Vec<Spec>, String> {
    let Value::Array(edges) = edges else {
        return Err("\"edges\" is a list".to_string());
    };
    let mut specs: Vec<Spec> = Vec::with_capacity(edges.len());
    let mut paths: HashSet<String> = HashSet::new();
    for (index, edge) in edges.into_iter().enumerate() {
        let line: EdgeLine =
            serde_json::from_value(edge).map_err(|e| format!("edge {index}: {e}"))?;
        let spec = check_edge(line)?;
        if specs.iter().any(|s| s.id == spec.id) {
            return Err(format!("edge {} is named twice", spec.id));
        }
        let from = match &spec.from {
            FromEnd::Pipe(path) => Some(path),
            FromEnd::Listen(_) => None,
        };
        let to = match &spec.to {
            ToEnd::Pipe(path) => Some(path),
            ToEnd::Dial { .. } => None,
        };
        for path in from.into_iter().chain(to) {
            if !paths.insert(path.clone()) {
                return Err(format!("edge {}: the pipe {path} is named twice", spec.id));
            }
        }
        specs.push(spec);
    }
    Ok(specs)
}

fn check_edge(line: EdgeLine) -> Result<Spec, String> {
    let id = line.id;
    if id.is_empty() {
        return Err("an edge with an empty id".to_string());
    }
    let end = |value: Value| {
        serde_json::from_value::<EndLine>(value).map_err(|_| {
            "an end is a pipe path, {\"listen\": KEY} or {\"dial\": [HOST, PORT], \"key\": KEY}"
                .to_string()
        })
    };
    let from = end(line.from)
        .and_then(EndLine::into_from)
        .map_err(|e| format!("edge {id}: from: {e}"))?;
    let to = end(line.to)
        .and_then(EndLine::into_to)
        .map_err(|e| format!("edge {id}: to: {e}"))?;
    // A spool holds whatever arrives, so a depth means nothing to it.
    let depth = match line.depth {
        _ if line.spool => DEFAULT_DEPTH,
        None => DEFAULT_DEPTH,
        Some(0) => return Err(format!("edge {id}: a depth of 0 would never read")),
        Some(depth) => usize::try_from(depth).map_err(|_| {
            format!("edge {id}: a depth of {depth} bytes is more than this machine holds")
        })?,
    };
    let buffer = match line.buffer {
        None => DEFAULT_BUFFER,
        Some(0) => return Err(format!("edge {id}: a pipe buffer of 0 bytes")),
        Some(buffer) => u32::try_from(buffer)
            .map_err(|_| format!("edge {id}: a pipe buffer of {buffer} bytes is past 4 GiB"))?,
    };
    Ok(Spec {
        id,
        from,
        to,
        depth,
        buffer,
        spool: line.spool,
    })
}

/// A lock that a panicking thread did not leave unusable: a panic is a
/// fault that ends the relay anyway, and the rows it writes on the way out
/// still need the edges.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn wait<'a, T>(condvar: &Condvar, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
    condvar.wait(guard).unwrap_or_else(PoisonError::into_inner)
}

/// One row on stderr, written whole while stderr is held, so two threads'
/// rows never interleave.
fn row(value: &impl Serialize) {
    let Ok(json) = serde_json::to_string(value) else {
        return;
    };
    let line = format!("{ROW_PREFIX}{json}\n");
    // A host that stopped reading stderr has gone; there is nobody to tell.
    let _ = io::stderr().lock().write_all(line.as_bytes());
}

/// A row about the relay rather than one edge's flow, `kind` first and
/// only the fields it has.
#[derive(Serialize, Default)]
struct Said<'a> {
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    ready: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    batch: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    edge: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    listening: Option<(&'a str, u16)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    refused: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'a str>,
}

fn say(said: Said) {
    row(&Said {
        kind: "relay",
        ..said
    });
}

/// The relay's fault, which it does not carry on from.
fn say_fault(error: &str) {
    say(Said {
        error: Some(error),
        ..Said::default()
    });
}

/// The relay's fault: its row, every FIFO unlinked, and the exit.
fn fault(message: &str) -> ! {
    say_fault(message);
    sys::unlink_all();
    std::process::exit(1);
}

#[derive(Serialize)]
struct FlowRow<'a> {
    kind: &'static str,
    edge: &'a str,
    #[serde(flatten)]
    shown: Shown,
}

/// What one flow row says, and what the last one said.
#[derive(Serialize, Clone, Copy, PartialEq, Eq, Default)]
struct Shown {
    moved: u64,
    began: bool,
    writing: bool,
    opening: bool,
    done: bool,
}

/// What changes an edge's row other than its counters.
#[derive(Default)]
struct FlowState {
    began: bool,
    source_open: bool,
    dest_open: bool,
    done: bool,
    shown: Shown,
}

/// How one end's blocking calls are broken off from another thread: the
/// end looks at `aborted` before each call, and an abort breaks off the
/// call it is already in, by the platform's means for a pipe and by a
/// shutdown for a socket. An abort that lands just before a call is
/// repeated by the report thread until the end lets go.
#[derive(Default)]
struct Gate {
    aborted: AtomicBool,
    pipe: sys::Hold,
    /// A second handle on the end's socket while it is open.
    socket: Mutex<Option<TcpStream>>,
}

impl Gate {
    fn aborted(&self) -> bool {
        self.aborted.load(Ordering::SeqCst)
    }

    fn abort(&self) {
        self.aborted.store(true, Ordering::SeqCst);
        self.nudge();
    }

    fn nudge(&self) {
        if !self.aborted() {
            return;
        }
        self.pipe.nudge();
        if let Some(socket) = &*lock(&self.socket) {
            let _ = socket.shutdown(Shutdown::Both);
        }
    }

    fn hold_socket(&self, socket: &TcpStream) {
        *lock(&self.socket) = socket.try_clone().ok();
        self.nudge();
    }

    fn release_socket(&self) {
        *lock(&self.socket) = None;
    }
}

/// An edge's source, opened.
enum Reader {
    Pipe(sys::Conn),
    Socket(TcpStream),
}

impl Reader {
    /// What has arrived, into `buf`; 0 once the stream has ended, by the
    /// producer closing, by an error, or by an abort.
    fn read(&mut self, buf: &mut [u8], gate: &Gate) -> usize {
        match self {
            Reader::Pipe(conn) => conn.read(buf),
            Reader::Socket(socket) => loop {
                if gate.aborted() {
                    return 0;
                }
                match socket.read(buf) {
                    Ok(n) => return n,
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(_) => return 0,
                }
            },
        }
    }

    fn close(self, gate: &Gate) {
        gate.release_socket();
        if let Reader::Pipe(conn) = self {
            conn.close();
        }
    }
}

/// An edge's destination, opened.
enum Writer {
    Pipe(sys::Conn),
    Socket(TcpStream),
}

impl Writer {
    fn write_all(&mut self, bytes: &[u8], gate: &Gate) -> io::Result<()> {
        if gate.aborted() {
            return Err(io::Error::other("the edge was stopped"));
        }
        match self {
            Writer::Pipe(conn) => conn.write_all(bytes),
            Writer::Socket(socket) => socket.write_all(bytes),
        }
    }

    /// The end of the stream, after its last byte: the consumer reads
    /// everything written, then the end of it.
    fn finish(self, gate: &Gate) {
        match self {
            Writer::Pipe(conn) => conn.finish(),
            Writer::Socket(socket) => {
                let _ = socket.shutdown(Shutdown::Write);
                gate.release_socket();
            }
        }
    }

    /// The end of the stream, cut off: whatever was not written is lost.
    fn close(self, gate: &Gate) {
        gate.release_socket();
        if let Writer::Pipe(conn) = self {
            conn.close();
        }
    }
}

/// A buffer of the queue, and how much of it holds bytes.
struct Chunk {
    buf: Box<[u8]>,
    len: usize,
}

fn new_buffer() -> Box<[u8]> {
    vec![0u8; CHUNK].into_boxed_slice()
}

/// What a spool keeps on disk once memory has held its share.
struct Spill {
    file: File,
    written: u64,
    read: u64,
}

impl Spill {
    /// A temporary file that is gone once it is closed, however the relay
    /// ends.
    fn new() -> io::Result<Spill> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir();
        loop {
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = dir.join(format!("ffrwd-relay-{}-{n}.spool", std::process::id()));
            let mut options = OpenOptions::new();
            options.read(true).write(true).create_new(true);
            #[cfg(windows)]
            {
                use std::os::windows::fs::OpenOptionsExt;
                options.custom_flags(
                    windows_sys::Win32::Storage::FileSystem::FILE_FLAG_DELETE_ON_CLOSE,
                );
            }
            match options.open(&path) {
                Ok(file) => {
                    #[cfg(unix)]
                    std::fs::remove_file(&path)?;
                    return Ok(Spill {
                        file,
                        written: 0,
                        read: 0,
                    });
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
    }

    fn append(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.file.seek(SeekFrom::Start(self.written))?;
        self.file.write_all(bytes)?;
        self.written += bytes.len() as u64;
        Ok(())
    }

    fn pending(&self) -> bool {
        self.read < self.written
    }

    /// The next stretch of the file, into `buf`: its length.
    fn take(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let left = usize::try_from(self.written - self.read).unwrap_or(usize::MAX);
        let len = buf.len().min(left);
        self.file.seek(SeekFrom::Start(self.read))?;
        self.file.read_exact(&mut buf[..len])?;
        self.read += len as u64;
        Ok(len)
    }
}

/// The bytes an edge has read and not yet handed on, and the buffers it
/// reuses for them.
#[derive(Default)]
struct Held {
    chunks: VecDeque<Chunk>,
    /// The bytes in `chunks`: what `depth` bounds.
    bytes: usize,
    pool: Vec<Box<[u8]>>,
    spill: Option<Spill>,
    /// The source has ended: nothing more will arrive.
    ended: bool,
    /// The edge is over: nothing more is handed on.
    stopped: bool,
}

impl Held {
    /// Queue `n` bytes of the reader's buffer. A read smaller than the room
    /// left in the last queued buffer is copied into it and the reader keeps
    /// its own, so a producer writing in small pieces never has the queue
    /// hold a mostly empty 64 KiB buffer for each.
    fn push(&mut self, buf: &mut Option<Box<[u8]>>, n: usize) {
        let Some(bytes) = buf.as_ref() else {
            return;
        };
        match self.chunks.back_mut() {
            Some(tail) if CHUNK - tail.len >= n => {
                tail.buf[tail.len..tail.len + n].copy_from_slice(&bytes[..n]);
                tail.len += n;
            }
            _ => {
                if let Some(buf) = buf.take() {
                    self.chunks.push_back(Chunk { buf, len: n });
                }
            }
        }
        self.bytes += n;
    }

    /// Hold `n` bytes for a spool: in memory to [`SPOOL_MEMORY`], and on disk
    /// from the first read that would pass it, as `_Spool` holds them.
    fn spool(&mut self, buf: &mut Option<Box<[u8]>>, n: usize) -> io::Result<()> {
        if self.spill.is_none() && self.bytes + n > SPOOL_MEMORY {
            self.spill = Some(Spill::new()?);
        }
        match (&mut self.spill, buf.as_ref()) {
            (Some(spill), Some(bytes)) => spill.append(&bytes[..n]),
            _ => {
                self.push(buf, n);
                Ok(())
            }
        }
    }

    /// Drop whatever is held, the spool's file with it.
    fn clear(&mut self) {
        while let Some(chunk) = self.chunks.pop_front() {
            self.pool.push(chunk.buf);
        }
        self.bytes = 0;
        self.spill = None;
    }
}

/// What the writer is handed next.
enum Took {
    Chunk(Chunk),
    /// The source ended and everything it sent has been handed on.
    End,
    /// The edge was stopped, or its other end failed.
    Stop,
    /// The spool's file could not be read back.
    Fail(io::Error),
}

/// One edge while it runs.
struct Edge {
    id: String,
    depth: usize,
    spool: bool,
    held: Mutex<Held>,
    /// The reader waits on this for room under `depth`.
    room: Condvar,
    /// The writer waits on this for bytes.
    arrived: Condvar,
    source: Arc<Gate>,
    dest: Arc<Gate>,
    moved: AtomicU64,
    writing: AtomicBool,
    state: Mutex<FlowState>,
    /// The edge's threads still running; the last one out ends the edge.
    running: AtomicUsize,
}

impl Edge {
    fn new(spec: &Spec, source: Arc<Gate>, dest: Arc<Gate>) -> Edge {
        Edge {
            id: spec.id.clone(),
            depth: spec.depth,
            spool: spec.spool,
            held: Mutex::new(Held::default()),
            room: Condvar::new(),
            arrived: Condvar::new(),
            source,
            dest,
            moved: AtomicU64::new(0),
            writing: AtomicBool::new(false),
            state: Mutex::new(FlowState::default()),
            running: AtomicUsize::new(2),
        }
    }

    /// Write this edge's row if it says something the last one did not.
    fn show(&self, state: &mut FlowState) {
        let shown = Shown {
            moved: self.moved.load(Ordering::SeqCst),
            began: state.began,
            writing: !state.done && self.writing.load(Ordering::SeqCst),
            opening: state.source_open && !state.dest_open && !state.done,
            done: state.done,
        };
        if shown != state.shown {
            state.shown = shown;
            row(&FlowRow {
                kind: "flow",
                edge: &self.id,
                shown,
            });
        }
    }

    /// Change the edge's state, and report the change at once.
    fn mark(&self, change: impl FnOnce(&mut FlowState)) {
        let mut state = lock(&self.state);
        change(&mut state);
        self.show(&mut state);
    }

    /// The periodic report: counters that moved, and a write that waits.
    fn report(&self) {
        self.show(&mut lock(&self.state));
    }

    /// End the edge now: drop what it holds, wake both threads, and break
    /// off whatever blocking call either end is in.
    fn stop(&self) {
        {
            let mut held = lock(&self.held);
            held.stopped = true;
            held.clear();
        }
        self.room.notify_all();
        self.arrived.notify_all();
        self.source.abort();
        self.dest.abort();
    }

    /// End the edge for a reason the host is told.
    fn fail(&self, why: &str) {
        say(Said {
            edge: Some(&self.id),
            error: Some(why),
            ..Said::default()
        });
        self.stop();
    }

    /// Repeat an abort that may have landed just before a blocking call.
    fn nudge(&self) {
        self.source.nudge();
        self.dest.nudge();
    }

    /// One of the edge's threads is through. The second ends the edge with
    /// its last row.
    fn leave(&self, relay: &Relay) {
        if self.running.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.writing.store(false, Ordering::SeqCst);
            self.mark(|state| state.done = true);
            relay.forget(&self.id);
        }
    }

    /// The reader's side: open the producer's end and read it until it ends.
    fn read_from(&self, source: Source, relay: &Relay) {
        let opened = match source {
            Source::Pipe(served) => {
                let path = served.path().to_string();
                served
                    .open()
                    .map(|conn| conn.map(Reader::Pipe))
                    .map_err(|e| format!("opening {path}: {e}"))
            }
            Source::Listen(key) => relay
                .cuts
                .take(&key, &self.source)
                .map(|socket| socket.map(Reader::Socket)),
        };
        match opened {
            Ok(Some(reader)) => {
                self.mark(|state| state.source_open = true);
                self.fill(reader);
            }
            Ok(None) => {}
            Err(why) => self.fail(&why),
        }
        lock(&self.held).ended = true;
        self.arrived.notify_all();
    }

    fn fill(&self, mut reader: Reader) {
        let mut buf: Option<Box<[u8]>> = None;
        let mut began = false;
        loop {
            {
                let mut held = lock(&self.held);
                while !held.stopped && !self.spool && held.bytes >= self.depth {
                    held = wait(&self.room, held);
                }
                if held.stopped {
                    break;
                }
                if buf.is_none() {
                    buf = held.pool.pop();
                }
            }
            let n = reader.read(buf.get_or_insert_with(new_buffer), &self.source);
            if n == 0 {
                break;
            }
            let mut held = lock(&self.held);
            if held.stopped {
                break;
            }
            if !self.spool {
                held.push(&mut buf, n);
                drop(held);
                self.arrived.notify_one();
                continue;
            }
            let spooled = held.spool(&mut buf, n);
            drop(held);
            if let Err(e) = spooled {
                self.fail(&format!("spooling to a temporary file: {e}"));
                break;
            }
            self.arrived.notify_one();
            // A spool counts what arrives: its consumer reads the document
            // whole when it opens, and until then this is the edge moving.
            self.moved.fetch_add(n as u64, Ordering::SeqCst);
            if !began {
                began = true;
                self.mark(|state| state.began = true);
            }
        }
        reader.close(&self.source);
    }

    /// The next chunk for the writer, returning the buffer it is done with.
    fn take(&self, spent: Option<Box<[u8]>>) -> Took {
        let mut held = lock(&self.held);
        if let Some(buf) = spent {
            held.pool.push(buf);
        }
        loop {
            if held.stopped {
                return Took::Stop;
            }
            if let Some(chunk) = held.chunks.pop_front() {
                held.bytes -= chunk.len;
                drop(held);
                self.room.notify_one();
                return Took::Chunk(chunk);
            }
            if held.spill.as_ref().is_some_and(Spill::pending) {
                let mut buf = held.pool.pop().unwrap_or_else(new_buffer);
                let Some(spill) = held.spill.as_mut() else {
                    continue;
                };
                return match spill.take(&mut buf) {
                    Ok(len) => Took::Chunk(Chunk { buf, len }),
                    Err(e) => Took::Fail(e),
                };
            }
            if held.ended {
                return Took::End;
            }
            held = wait(&self.arrived, held);
        }
    }

    /// The writer's side: open the consumer's end and hand it every chunk.
    fn write_to(&self, dest: Dest, relay: &Relay) {
        let opened = match dest {
            Dest::Pipe(served) => {
                let path = served.path().to_string();
                served
                    .open()
                    .map(|conn| conn.map(Writer::Pipe))
                    .map_err(|e| format!("opening {path}: {e}"))
            }
            Dest::Dial { host, port, key } => {
                dial(&host, port, &key, relay.secret.as_deref(), &self.dest)
                    .map(|socket| socket.map(Writer::Socket))
            }
        };
        match opened {
            Ok(Some(writer)) => {
                self.mark(|state| state.dest_open = true);
                self.drain(writer);
            }
            Ok(None) => {}
            Err(why) => self.fail(&why),
        }
    }

    fn drain(&self, mut writer: Writer) {
        let mut spent: Option<Box<[u8]>> = None;
        let mut began = false;
        loop {
            match self.take(spent.take()) {
                Took::Chunk(chunk) => {
                    self.writing.store(true, Ordering::SeqCst);
                    let wrote = writer.write_all(&chunk.buf[..chunk.len], &self.dest);
                    self.writing.store(false, Ordering::SeqCst);
                    if wrote.is_err() {
                        // The consumer is gone. Closing the source is what
                        // tells the producer, with the broken pipe it has
                        // always been told by.
                        self.stop();
                        writer.close(&self.dest);
                        return;
                    }
                    if !self.spool {
                        self.moved.fetch_add(chunk.len as u64, Ordering::SeqCst);
                        if !began {
                            began = true;
                            self.mark(|state| state.began = true);
                        }
                    }
                    spent = Some(chunk.buf);
                }
                Took::End => {
                    writer.finish(&self.dest);
                    return;
                }
                Took::Stop => {
                    writer.close(&self.dest);
                    return;
                }
                Took::Fail(e) => {
                    self.fail(&format!("reading back the spool: {e}"));
                    writer.close(&self.dest);
                    return;
                }
            }
        }
    }
}

/// An edge's source before it opens.
enum Source {
    Pipe(sys::Served),
    Listen(String),
}

/// An edge's destination before it opens.
enum Dest {
    Pipe(sys::Served),
    Dial {
        host: String,
        port: u16,
        key: String,
    },
}

/// Reach the node listening for `key`, retrying while it refuses (it may not
/// be listening yet), and open the edge with the job's secret. None once the
/// end was aborted.
fn dial(
    host: &str,
    port: u16,
    key: &str,
    secret: Option<&str>,
    gate: &Gate,
) -> Result<Option<TcpStream>, String> {
    let Some(secret) = secret else {
        return Err(format!(
            "{SECRET_ENV} is not set, so there is no secret to open {key} with"
        ));
    };
    let addresses: Vec<SocketAddr> = (host, port)
        .to_socket_addrs()
        .map_err(|e| format!("finding {host}:{port}: {e}"))?
        .collect();
    let Some(address) = addresses.first().copied() else {
        return Err(format!("{host}:{port} names no address"));
    };
    let mut socket = loop {
        if gate.aborted() {
            return Ok(None);
        }
        match TcpStream::connect_timeout(&address, DIAL_WAIT) {
            Ok(socket) => break socket,
            Err(e)
                if matches!(
                    e.kind(),
                    io::ErrorKind::ConnectionRefused | io::ErrorKind::TimedOut
                ) =>
            {
                thread::sleep(DIAL_AGAIN)
            }
            Err(e) => return Err(format!("dialing {host}:{port} for {key}: {e}")),
        }
    };
    gate.hold_socket(&socket);
    let _ = socket.set_nodelay(true);
    let refused = |e: io::Error| {
        format!("the node at {host}:{port} did not take {key}: {e} (a wrong secret, or a key it does not listen for, is closed unanswered)")
    };
    socket
        .write_all(format!("{CUT_PROTOCOL} {secret} {key}\n").as_bytes())
        .map_err(refused)?;
    let mut answer = [0u8; 3];
    socket.read_exact(&mut answer).map_err(refused)?;
    if &answer != b"OK\n" {
        return Err(format!(
            "the node at {host}:{port} answered {:?} for {key}, not OK",
            String::from_utf8_lossy(&answer)
        ));
    }
    if gate.aborted() {
        return Ok(None);
    }
    Ok(Some(socket))
}

/// Where a cut edge's connection stands on the listening node.
enum Arrival {
    /// Listened for, and nothing has come yet.
    Waiting,
    /// A connection has passed and is being answered.
    Answering,
    /// Answered, and waiting for its edge to take it.
    Parked(TcpStream),
    /// Its edge has it.
    Taken,
}

/// The node's data port: the keys it listens for and what has arrived.
#[derive(Default)]
struct Cuts {
    keys: Mutex<HashMap<String, Arrival>>,
    arrived: Condvar,
}

impl Cuts {
    /// The connection for `key`, once it has arrived. None once the end was
    /// aborted.
    fn take(&self, key: &str, gate: &Gate) -> Result<Option<TcpStream>, String> {
        let mut keys = lock(&self.keys);
        loop {
            if gate.aborted() {
                return Ok(None);
            }
            match keys.get_mut(key) {
                None => return Err(format!("{key} is not a key this node listens for")),
                Some(Arrival::Taken) => return Err(format!("{key} is taken by another edge")),
                Some(arrival @ Arrival::Parked(_)) => {
                    let Arrival::Parked(socket) = std::mem::replace(arrival, Arrival::Taken) else {
                        continue;
                    };
                    drop(keys);
                    gate.hold_socket(&socket);
                    return Ok(Some(socket));
                }
                Some(Arrival::Waiting | Arrival::Answering) => {
                    keys = self
                        .arrived
                        .wait_timeout(keys, ARRIVAL_POLL)
                        .unwrap_or_else(PoisonError::into_inner)
                        .0;
                }
            }
        }
    }

    /// Answer one connection to the data port: take it for its key, or close
    /// it and say why.
    fn answer(&self, mut socket: TcpStream, secret: &str) {
        let key = match self.admit(&mut socket, secret) {
            Ok(key) => key,
            Err(reason) => {
                drop(socket);
                say(Said {
                    refused: Some(&reason),
                    ..Said::default()
                });
                return;
            }
        };
        if let Some(arrival) = lock(&self.keys).get_mut(&key) {
            *arrival = Arrival::Parked(socket);
        }
        self.arrived.notify_all();
    }

    /// Read the opening line and answer it: the key the connection is for,
    /// held for it as `Answering` until it is parked, or why it is refused.
    fn admit(&self, socket: &mut TcpStream, secret: &str) -> Result<String, String> {
        let line = opening_line(socket)?;
        let line = String::from_utf8_lossy(&line);
        let words: Vec<&str> = line.split_whitespace().collect();
        if words.len() != 4 || format!("{} {}", words[0], words[1]) != CUT_PROTOCOL {
            return Err("not a cut edge's opening line".to_string());
        }
        if !same(words[2].as_bytes(), secret.as_bytes()) {
            return Err("the wrong secret".to_string());
        }
        let key = words[3];
        match lock(&self.keys).get_mut(key) {
            None => return Err(format!("an edge this node does not listen for: {key}")),
            Some(arrival @ Arrival::Waiting) => *arrival = Arrival::Answering,
            Some(_) => return Err(format!("a second connection for {key}")),
        }
        let answered = socket
            .write_all(b"OK\n")
            .and_then(|()| socket.set_read_timeout(None));
        if answered.is_err() {
            // The key is free again for the node to dial once more.
            if let Some(arrival) = lock(&self.keys).get_mut(key) {
                *arrival = Arrival::Waiting;
            }
            return Err("gone before it was answered".to_string());
        }
        Ok(key.to_string())
    }
}

/// A connection's opening line, its newline included, read a byte at a time
/// so nothing past it is taken off the socket: the edge's bytes follow it.
fn opening_line(socket: &mut TcpStream) -> Result<Vec<u8>, String> {
    let deadline = Instant::now() + OPENING_WAIT;
    let mut line = Vec::with_capacity(OPENING_MOST);
    let mut byte = [0u8; 1];
    loop {
        if line.len() == OPENING_MOST {
            return Err("an opening line too long".to_string());
        }
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() || socket.set_read_timeout(Some(left)).is_err() {
            return Err("no opening line in time".to_string());
        }
        match socket.read(&mut byte) {
            Ok(0) => return Err("closed before its opening line".to_string()),
            Ok(_) => {
                line.push(byte[0]);
                if byte[0] == b'\n' {
                    return Ok(line);
                }
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e)
                if matches!(
                    e.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                return Err("no opening line in time".to_string())
            }
            Err(_) => return Err("closed before its opening line".to_string()),
        }
    }
}

/// Whether two secrets are the same, in a time that does not say how much
/// of one matched.
fn same(given: &[u8], secret: &[u8]) -> bool {
    let mut differ = given.len() ^ secret.len();
    for i in 0..given.len().max(secret.len()) {
        let a = given.get(i).copied().unwrap_or(0);
        let b = secret.get(i).copied().unwrap_or(0);
        differ |= usize::from(a ^ b);
    }
    differ == 0
}

/// The address a data port binds: IPv6 when the host is written as one,
/// otherwise the host's first IPv4 address.
fn port_address(host: &str) -> Result<SocketAddr, String> {
    if host.contains(':') {
        let ip: Ipv6Addr = host
            .trim_start_matches('[')
            .trim_end_matches(']')
            .parse()
            .map_err(|e| format!("{host} is not an IPv6 address: {e}"))?;
        return Ok(SocketAddr::from((ip, 0)));
    }
    (host, 0)
        .to_socket_addrs()
        .map_err(|e| format!("finding {host}: {e}"))?
        .find(SocketAddr::is_ipv4)
        .ok_or_else(|| format!("{host} has no IPv4 address"))
}

/// Every edge still running, by id, and the node's data port.
#[derive(Default)]
struct Relay {
    edges: Mutex<HashMap<String, Arc<Edge>>>,
    emptied: Condvar,
    cuts: Arc<Cuts>,
    secret: Option<String>,
}

impl Relay {
    fn running(&self) -> Vec<Arc<Edge>> {
        lock(&self.edges).values().cloned().collect()
    }

    fn forget(&self, id: &str) {
        let mut edges = lock(&self.edges);
        edges.remove(id);
        if edges.is_empty() {
            self.emptied.notify_all();
        }
    }

    /// Make every pipe of a batch and start its edges. Every serving end is
    /// past the point of starting its wait for a client when this returns,
    /// so the members the host spawns next find their pipes listening.
    fn start(self: &Arc<Self>, specs: Vec<Spec>) -> Result<(), String> {
        {
            let edges = lock(&self.edges);
            if let Some(spec) = specs.iter().find(|s| edges.contains_key(&s.id)) {
                return Err(format!("edge {} is already running", spec.id));
            }
        }
        // Every pipe first: a batch that cannot make one starts nothing, and
        // the pipes it did make go as the half-built list is dropped.
        let mut made = Vec::with_capacity(specs.len());
        for spec in specs {
            let source = Arc::new(Gate::default());
            let dest = Arc::new(Gate::default());
            let from = match &spec.from {
                FromEnd::Pipe(path) => Source::Pipe(
                    sys::Served::create(path, true, spec.buffer, Arc::clone(&source))
                        .map_err(|e| format!("edge {}: making {path}: {e}", spec.id))?,
                ),
                FromEnd::Listen(key) => Source::Listen(key.clone()),
            };
            let to = match &spec.to {
                ToEnd::Pipe(path) => Dest::Pipe(
                    sys::Served::create(path, false, spec.buffer, Arc::clone(&dest))
                        .map_err(|e| format!("edge {}: making {path}: {e}", spec.id))?,
                ),
                ToEnd::Dial { host, port, key } => Dest::Dial {
                    host: host.clone(),
                    port: *port,
                    key: key.clone(),
                },
            };
            made.push((Arc::new(Edge::new(&spec, source, dest)), from, to));
        }

        let (armed, listening) = mpsc::channel::<()>();
        for (edge, from, to) in made {
            lock(&self.edges).insert(edge.id.clone(), Arc::clone(&edge));
            let relay = Arc::clone(self);
            let reading = Arc::clone(&edge);
            let ready = armed.clone();
            spawn_or_fault(format!("relay-read-{}", edge.id), move || {
                drop(ready);
                reading.read_from(from, &relay);
                reading.leave(&relay);
            });
            let relay = Arc::clone(self);
            let writing = Arc::clone(&edge);
            let ready = armed.clone();
            spawn_or_fault(format!("relay-write-{}", edge.id), move || {
                drop(ready);
                writing.write_to(to, &relay);
                writing.leave(&relay);
            });
        }
        // Each thread lets go of its sender just before its end starts
        // waiting for a client; the channel closes when the last one has.
        drop(armed);
        while listening.recv().is_ok() {}
        Ok(())
    }

    fn stop(&self, ids: &[String]) {
        let edges: Vec<Arc<Edge>> = {
            let running = lock(&self.edges);
            ids.iter()
                .filter_map(|id| running.get(id).cloned())
                .collect()
        };
        for edge in edges {
            edge.stop();
        }
    }

    /// Open the data port on `host` for `keys`: the port it listens on.
    fn listen(&self, host: &str, keys: Vec<String>) -> Result<u16, String> {
        let Some(secret) = self.secret.clone() else {
            return Err(format!(
                "{SECRET_ENV} is not set, and a data port takes no connection without the job's secret"
            ));
        };
        let listener = TcpListener::bind(port_address(host)?)
            .map_err(|e| format!("opening a data port on {host}: {e}"))?;
        let port = listener
            .local_addr()
            .map_err(|e| format!("opening a data port on {host}: {e}"))?
            .port();
        {
            let mut listened = lock(&self.cuts.keys);
            for key in keys {
                listened.entry(key).or_insert(Arrival::Waiting);
            }
        }
        let cuts = Arc::clone(&self.cuts);
        spawn_or_fault(format!("relay-listen-{port}"), move || {
            for socket in listener.incoming() {
                let Ok(socket) = socket else {
                    // A connection that failed before it was accepted; the
                    // pause keeps a port in trouble from spinning.
                    thread::sleep(DIAL_AGAIN);
                    continue;
                };
                let cuts = Arc::clone(&cuts);
                let secret = secret.clone();
                // One that cannot be given a thread is closed unanswered,
                // and its node dials again.
                let _ = spawn("relay-answer".to_string(), move || {
                    cuts.answer(socket, &secret)
                });
            }
        });
        Ok(port)
    }

    /// Abort every edge, give them a moment to let go of their ends, and
    /// unlink whatever FIFO is still there.
    fn shut(&self, code: i32) -> i32 {
        for edge in self.running() {
            edge.stop();
        }
        let deadline = Instant::now() + SHUTDOWN_GRACE;
        let mut edges = lock(&self.edges);
        while !edges.is_empty() {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break;
            }
            edges = self
                .emptied
                .wait_timeout(edges, left)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        drop(edges);
        sys::unlink_all();
        code
    }
}

/// A thread of the relay. One that panics is a fault of the whole relay: an
/// edge it leaves half run would never end.
fn spawn(name: String, body: impl FnOnce() + Send + 'static) -> io::Result<()> {
    thread::Builder::new()
        .name(name.clone())
        .spawn(move || {
            if catch_unwind(AssertUnwindSafe(body)).is_err() {
                fault(&format!("thread {name} panicked"));
            }
        })
        .map(drop)
}

/// A thread the relay cannot run without.
fn spawn_or_fault(name: String, body: impl FnOnce() + Send + 'static) {
    if let Err(e) = spawn(name, body) {
        fault(&format!("starting a thread: {e}"));
    }
}

/// `ffrwd-wasm relay`: serve the edges stdin names until stdin closes. The
/// exit code: 0 once stdin has closed, 1 for a fault, 2 for an argv the
/// relay does not take.
pub fn main(args: &[String]) -> i32 {
    if let Some(arg) = args.first() {
        say_fault(&format!(
            "relay takes no arguments, and was given {arg}: its edges arrive on stdin"
        ));
        return 2;
    }
    let relay = Arc::new(Relay {
        secret: std::env::var(SECRET_ENV).ok().filter(|s| !s.is_empty()),
        ..Relay::default()
    });

    let watched = Arc::clone(&relay);
    spawn_or_fault("relay-report".to_string(), move || loop {
        thread::sleep(REPORT_EVERY);
        for edge in watched.running() {
            edge.report();
            edge.nudge();
        }
    });

    for line in io::stdin().lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(e) => {
                say_fault(&format!("reading stdin: {e}"));
                return relay.shut(1);
            }
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match parse_command(line) {
            Ok(Command::Batch(batch, specs)) => match specs.and_then(|specs| relay.start(specs)) {
                Ok(()) => say(Said {
                    ready: Some(batch),
                    ..Said::default()
                }),
                Err(error) => say(Said {
                    batch: Some(batch),
                    error: Some(&error),
                    ..Said::default()
                }),
            },
            Ok(Command::Stop(ids)) => relay.stop(&ids),
            // A data port that cannot be opened is said, and the relay
            // carries on: the host fails the node, not this process.
            Ok(Command::Listen(listen)) => {
                match listen.and_then(|(host, keys)| relay.listen(&host, keys).map(|p| (host, p))) {
                    Ok((host, port)) => say(Said {
                        listening: Some((&host, port)),
                        ..Said::default()
                    }),
                    Err(error) => say_fault(&error),
                }
            }
            Err(error) => {
                say_fault(&error);
                return relay.shut(1);
            }
        }
    }
    relay.shut(0)
}

/// Windows named pipes: `CreateNamedPipeW` makes one, `ConnectNamedPipe`
/// waits for its client, and the handle is a synchronous one the relay
/// reads or writes as a file. `CancelIoEx` breaks off whatever the handle
/// is waiting in, a connect, a read, a write or a flush.
#[cfg(windows)]
mod sys {
    use std::ffi::OsStr;
    use std::fs::File;
    use std::io::{self, Read, Write};
    use std::iter::once;
    use std::os::windows::ffi::OsStrExt;
    use std::os::windows::io::{AsRawHandle, FromRawHandle, RawHandle};
    use std::ptr::{null, null_mut};
    use std::sync::{Arc, Mutex};

    use windows_sys::Win32::Foundation::{
        ERROR_NO_DATA, ERROR_PIPE_CONNECTED, HANDLE, INVALID_HANDLE_VALUE,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        FlushFileBuffers, PIPE_ACCESS_INBOUND, PIPE_ACCESS_OUTBOUND,
    };
    use windows_sys::Win32::System::Pipes::{
        ConnectNamedPipe, CreateNamedPipeW, PIPE_TYPE_BYTE, PIPE_WAIT,
    };
    use windows_sys::Win32::System::IO::CancelIoEx;

    use super::{lock, Gate};

    /// An end's pipe handle while the end holds it open, as a number: a
    /// handle closed and reused is never cancelled by mistake, since the end
    /// takes it out of here before closing it.
    #[derive(Default)]
    pub struct Hold {
        handle: Mutex<Option<usize>>,
    }

    impl Hold {
        pub fn nudge(&self) {
            if let Some(handle) = *lock(&self.handle) {
                // SAFETY: the handle is open while it is held here.
                unsafe { CancelIoEx(handle as HANDLE, null()) };
            }
        }

        fn hold(&self, handle: RawHandle) {
            *lock(&self.handle) = Some(handle as usize);
        }

        fn release(&self) {
            *lock(&self.handle) = None;
        }
    }

    /// A pipe end and the gate that can break off its calls. The gate lets
    /// go of the handle before the file closes it.
    struct Pipe {
        file: File,
        path: String,
        gate: Arc<Gate>,
    }

    impl Drop for Pipe {
        fn drop(&mut self) {
            self.gate.pipe.release();
        }
    }

    impl Pipe {
        fn raw(&self) -> HANDLE {
            self.file.as_raw_handle() as HANDLE
        }
    }

    /// A pipe made and not yet opened by its client.
    pub struct Served(Pipe);

    impl Served {
        /// One instance, byte mode, blocking calls, `buffer` bytes each way:
        /// what `pipes.create` makes today. `reads` is the relay's direction.
        pub fn create(path: &str, reads: bool, buffer: u32, gate: Arc<Gate>) -> io::Result<Served> {
            let wide: Vec<u16> = OsStr::new(path).encode_wide().chain(once(0)).collect();
            let access = if reads {
                PIPE_ACCESS_INBOUND
            } else {
                PIPE_ACCESS_OUTBOUND
            };
            // SAFETY: `wide` is NUL terminated and outlives the call.
            let handle = unsafe {
                CreateNamedPipeW(
                    wide.as_ptr(),
                    access,
                    PIPE_TYPE_BYTE | PIPE_WAIT,
                    1,
                    buffer,
                    buffer,
                    0,
                    null(),
                )
            };
            if handle == INVALID_HANDLE_VALUE {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: a handle just made, owned by nothing else.
            let file = unsafe { File::from_raw_handle(handle as RawHandle) };
            gate.pipe.hold(file.as_raw_handle());
            Ok(Served(Pipe {
                file,
                path: path.to_string(),
                gate,
            }))
        }

        pub fn path(&self) -> &str {
            &self.0.path
        }

        /// Wait for the client. None once the end was aborted.
        pub fn open(self) -> io::Result<Option<Conn>> {
            let pipe = self.0;
            if pipe.gate.aborted() {
                return Ok(None);
            }
            // SAFETY: the handle is open for as long as `pipe` is.
            if unsafe { ConnectNamedPipe(pipe.raw(), null_mut()) } == 0 {
                let e = io::Error::last_os_error();
                let code = e.raw_os_error().unwrap_or(0) as u32;
                // A client that opened before the call, or opened and has
                // already closed again, is a client all the same.
                if code != ERROR_PIPE_CONNECTED && code != ERROR_NO_DATA {
                    return if pipe.gate.aborted() {
                        Ok(None)
                    } else {
                        Err(e)
                    };
                }
            }
            if pipe.gate.aborted() {
                return Ok(None);
            }
            Ok(Some(Conn(pipe)))
        }
    }

    /// A pipe its client has opened.
    pub struct Conn(Pipe);

    impl Conn {
        /// What has arrived, into `buf`; 0 once the stream has ended, by the
        /// producer closing, by an error, or by an abort.
        pub fn read(&mut self, buf: &mut [u8]) -> usize {
            loop {
                if self.0.gate.aborted() {
                    return 0;
                }
                match self.0.file.read(buf) {
                    Ok(n) => return n,
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(_) => return 0,
                }
            }
        }

        pub fn write_all(&mut self, bytes: &[u8]) -> io::Result<()> {
            self.0.file.write_all(bytes)
        }

        /// The end of the stream: wait for the client to read everything,
        /// then close. `NamedPipe.close` called `DisconnectNamedPipe` between
        /// the two, and that is not done here: a client's next read after a
        /// disconnect fails with ERROR_PIPE_NOT_CONNECTED, which Python reads
        /// as EINVAL and ffmpeg logs as "Error during demuxing: Invalid
        /// argument", where a plain close is ERROR_BROKEN_PIPE, the end of
        /// the stream to both.
        pub fn finish(self) {
            if !self.0.gate.aborted() {
                // SAFETY: the handle is open for as long as `self` is.
                unsafe {
                    FlushFileBuffers(self.0.raw());
                }
            }
        }

        pub fn close(self) {}
    }

    /// Windows removes a named pipe with its last handle.
    pub fn unlink_all() {}
}

/// POSIX FIFOs: `mkfifo` makes one, and opening it is what waits for the
/// process on the other end. Once open, every read and write waits in
/// `poll`, a tenth of a second at a time, so an abort is seen without a
/// signal or a second descriptor per end.
#[cfg(unix)]
mod sys {
    use std::ffi::CString;
    use std::io;
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use super::{lock, Gate};

    /// How long one wait in `poll` lasts before the end looks for an abort.
    const POLL_MS: libc::c_int = 100;

    /// How often a writing end tries a FIFO nobody has opened for reading.
    const OPEN_POLL: Duration = Duration::from_millis(10);

    /// The FIFOs made and not yet unlinked, for an exit that cannot wait for
    /// their ends to close.
    static FIFOS: Mutex<Vec<CString>> = Mutex::new(Vec::new());

    fn unlink(path: &CString) {
        // SAFETY: a NUL terminated path.
        unsafe { libc::unlink(path.as_ptr()) };
        lock(&FIFOS).retain(|p| p != path);
    }

    pub fn unlink_all() {
        for path in std::mem::take(&mut *lock(&FIFOS)) {
            // SAFETY: a NUL terminated path.
            unsafe { libc::unlink(path.as_ptr()) };
        }
    }

    /// A reading end still in its `open` waits for a writer, so an abort
    /// opens the FIFO itself, read and write, and holds it open until the
    /// end lets go: that is the writer the `open` was waiting for, whenever
    /// it begins.
    #[derive(Default)]
    pub struct Hold {
        opening: Mutex<Opening>,
    }

    #[derive(Default)]
    struct Opening {
        path: Option<CString>,
        holder: Option<OwnedFd>,
    }

    impl Hold {
        pub fn nudge(&self) {
            let mut opening = lock(&self.opening);
            if opening.holder.is_some() {
                return;
            }
            if let Some(path) = &opening.path {
                // SAFETY: a NUL terminated path.
                let fd = unsafe {
                    libc::open(
                        path.as_ptr(),
                        libc::O_RDWR | libc::O_NONBLOCK | libc::O_CLOEXEC,
                    )
                };
                if fd >= 0 {
                    // SAFETY: a descriptor just opened, owned by nothing else.
                    opening.holder = Some(unsafe { OwnedFd::from_raw_fd(fd) });
                }
            }
        }

        fn waiting(&self, path: &CString) {
            lock(&self.opening).path = Some(path.clone());
        }

        fn release(&self) {
            let mut opening = lock(&self.opening);
            opening.path = None;
            opening.holder = None;
        }
    }

    /// A FIFO, its descriptor once open, and the gate that can break off its
    /// calls. It is unlinked when it closes.
    struct Pipe {
        fd: Option<OwnedFd>,
        path: CString,
        display: String,
        buffer: u32,
        reads: bool,
        gate: Arc<Gate>,
    }

    impl Drop for Pipe {
        fn drop(&mut self) {
            self.gate.pipe.release();
            unlink(&self.path);
        }
    }

    /// A FIFO made and not yet opened by the process on its other end.
    pub struct Served(Pipe);

    impl Served {
        /// `mkfifo 0600`, as `pipes.create` makes one. `reads` is the
        /// relay's direction; `buffer` is asked of the kernel once it is open.
        pub fn create(path: &str, reads: bool, buffer: u32, gate: Arc<Gate>) -> io::Result<Served> {
            let c_path = CString::new(path).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidInput, "a path with a NUL in it")
            })?;
            // SAFETY: a NUL terminated path.
            if unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) } != 0 {
                return Err(io::Error::last_os_error());
            }
            lock(&FIFOS).push(c_path.clone());
            Ok(Served(Pipe {
                fd: None,
                path: c_path,
                display: path.to_string(),
                buffer,
                reads,
                gate,
            }))
        }

        pub fn path(&self) -> &str {
            &self.0.display
        }

        /// Wait for the process on the other end. None once the end was
        /// aborted.
        pub fn open(self) -> io::Result<Option<Conn>> {
            let mut pipe = self.0;
            let fd = if pipe.reads {
                pipe.gate.pipe.waiting(&pipe.path);
                if pipe.gate.aborted() {
                    return Ok(None);
                }
                loop {
                    // SAFETY: a NUL terminated path.
                    let fd =
                        unsafe { libc::open(pipe.path.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC) };
                    if fd >= 0 {
                        break fd;
                    }
                    let e = io::Error::last_os_error();
                    if e.kind() != io::ErrorKind::Interrupted {
                        return Err(e);
                    }
                }
            } else {
                loop {
                    if pipe.gate.aborted() {
                        return Ok(None);
                    }
                    // SAFETY: a NUL terminated path.
                    let fd = unsafe {
                        libc::open(
                            pipe.path.as_ptr(),
                            libc::O_WRONLY | libc::O_NONBLOCK | libc::O_CLOEXEC,
                        )
                    };
                    if fd >= 0 {
                        break fd;
                    }
                    let e = io::Error::last_os_error();
                    match e.raw_os_error() {
                        // Nobody has opened it for reading yet.
                        Some(libc::ENXIO) => std::thread::sleep(OPEN_POLL),
                        Some(libc::EINTR) => {}
                        _ => return Err(e),
                    }
                }
            };
            // SAFETY: a descriptor just opened, owned by nothing else.
            let fd = unsafe { OwnedFd::from_raw_fd(fd) };
            if pipe.gate.aborted() {
                return Ok(None);
            }
            set_nonblocking(&fd)?;
            set_size(&fd, pipe.buffer);
            pipe.fd = Some(fd);
            Ok(Some(Conn(pipe)))
        }
    }

    fn set_nonblocking(fd: &OwnedFd) -> io::Result<()> {
        // SAFETY: an open descriptor.
        let flags = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GETFL) };
        // SAFETY: likewise.
        if flags < 0
            || unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// Ask Linux for a pipe of `buffer` bytes, no more than
    /// `/proc/sys/fs/pipe-max-size` allows (a larger ask is refused outright,
    /// not shortened). Best effort, as `pipes._set_size` is: the pipe carries
    /// its stream at whatever size it ends up.
    #[cfg(target_os = "linux")]
    fn set_size(fd: &OwnedFd, buffer: u32) {
        if buffer <= super::DEFAULT_BUFFER {
            return;
        }
        let most = std::fs::read_to_string("/proc/sys/fs/pipe-max-size")
            .ok()
            .and_then(|text| text.trim().parse::<u32>().ok())
            .unwrap_or(buffer);
        let size = libc::c_int::try_from(buffer.min(most)).unwrap_or(libc::c_int::MAX);
        // SAFETY: an open descriptor.
        unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETPIPE_SZ, size) };
    }

    /// Every other POSIX sizes its FIFOs itself.
    #[cfg(not(target_os = "linux"))]
    fn set_size(_fd: &OwnedFd, _buffer: u32) {}

    /// Wait until `fd` is ready for `events`, or a tenth of a second has
    /// passed.
    fn wait_for(fd: &OwnedFd, events: libc::c_short) {
        let mut poll = libc::pollfd {
            fd: fd.as_raw_fd(),
            events,
            revents: 0,
        };
        // SAFETY: one pollfd, alive for the call.
        unsafe { libc::poll(&mut poll, 1, POLL_MS) };
    }

    /// A FIFO the process on the other end has opened.
    pub struct Conn(Pipe);

    impl Conn {
        fn raw(&self) -> libc::c_int {
            self.0.fd.as_ref().map_or(-1, AsRawFd::as_raw_fd)
        }

        /// What has arrived, into `buf`; 0 once the stream has ended, by the
        /// producer closing, by an error, or by an abort.
        pub fn read(&mut self, buf: &mut [u8]) -> usize {
            let Some(fd) = self.0.fd.as_ref() else {
                return 0;
            };
            loop {
                if self.0.gate.aborted() {
                    return 0;
                }
                wait_for(fd, libc::POLLIN);
                // SAFETY: `buf` is valid for its length.
                let n = unsafe { libc::read(self.raw(), buf.as_mut_ptr().cast(), buf.len()) };
                if n >= 0 {
                    return n as usize;
                }
                match io::Error::last_os_error().raw_os_error() {
                    Some(libc::EAGAIN | libc::EINTR) => continue,
                    _ => return 0,
                }
            }
        }

        pub fn write_all(&mut self, mut bytes: &[u8]) -> io::Result<()> {
            let Some(fd) = self.0.fd.as_ref() else {
                return Err(io::Error::from(io::ErrorKind::NotConnected));
            };
            while !bytes.is_empty() {
                if self.0.gate.aborted() {
                    return Err(io::Error::other("the edge was stopped"));
                }
                wait_for(fd, libc::POLLOUT);
                // SAFETY: `bytes` is valid for its length.
                let n = unsafe { libc::write(self.raw(), bytes.as_ptr().cast(), bytes.len()) };
                if n >= 0 {
                    bytes = &bytes[n as usize..];
                    continue;
                }
                let e = io::Error::last_os_error();
                match e.raw_os_error() {
                    Some(libc::EAGAIN | libc::EINTR) => continue,
                    _ => return Err(e),
                }
            }
            Ok(())
        }

        /// The end of the stream: close and unlink. What was written stays
        /// readable after the close.
        pub fn finish(self) {}

        pub fn close(self) {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch(line: &str) -> (u64, Result<Vec<Spec>, String>) {
        match parse_command(line) {
            Ok(Command::Batch(n, specs)) => (n, specs),
            Ok(_) => panic!("{line} parsed as something else"),
            Err(e) => panic!("{line} refused: {e}"),
        }
    }

    #[test]
    fn a_batch_takes_its_defaults_and_its_cut_ends() {
        let (n, specs) = batch(
            r#"{"batch": 3, "edges": [
                {"id": "e", "from": "a", "to": "b"},
                {"id": "f", "from": {"listen": "k1"}, "to": {"dial": ["10.0.0.2", 4000], "key": "k2"},
                 "depth": 1048576, "buffer": 65536, "spool": false},
                {"id": "g", "from": "c", "to": "d", "depth": 0, "spool": true}
            ]}"#,
        );
        let specs = specs.expect("a good batch");
        assert_eq!(n, 3);
        assert_eq!(specs[0].depth, DEFAULT_DEPTH);
        assert_eq!(specs[0].buffer, DEFAULT_BUFFER);
        assert!(!specs[0].spool);
        assert_eq!(specs[1].from, FromEnd::Listen("k1".to_string()));
        assert_eq!(
            specs[1].to,
            ToEnd::Dial {
                host: "10.0.0.2".to_string(),
                port: 4000,
                key: "k2".to_string()
            }
        );
        assert_eq!(specs[1].depth, 1 << 20);
        assert!(specs[2].spool, "a spool's depth is no business of its own");
    }

    #[test]
    fn a_bad_edge_is_the_batchs_error_and_not_the_relays() {
        for (edges, says) in [
            (r#"[{"id": "e", "from": "a"}]"#, "missing field `to`"),
            (
                r#"[{"id": "e", "from": "a", "to": "b", "deep": 1}]"#,
                "unknown field `deep`",
            ),
            (
                r#"[{"id": "e", "from": "a", "to": "b", "depth": 0}]"#,
                "depth of 0",
            ),
            (
                r#"[{"id": "e", "from": "a", "to": "b", "buffer": 0}]"#,
                "buffer of 0",
            ),
            (r#"[{"id": "e", "from": "a", "to": "a"}]"#, "named twice"),
            (r#"[{"id": "e", "from": "", "to": "b"}]"#, "empty"),
            (
                r#"[{"id": "e", "from": 7, "to": "b"}]"#,
                "an end is a pipe path",
            ),
            (
                r#"[{"id": "e", "from": "a", "to": {"listen": "k"}}]"#,
                "a \"from\" and not a \"to\"",
            ),
            (
                r#"[{"id": "e", "from": {"dial": ["h", 1], "key": "k"}, "to": "b"}]"#,
                "a \"to\" and not a \"from\"",
            ),
            (
                r#"[{"id": "e", "from": "a", "to": "b"}, {"id": "e", "from": "c", "to": "d"}]"#,
                "edge e is named twice",
            ),
        ] {
            let (n, specs) = batch(&format!(r#"{{"batch": 7, "edges": {edges}}}"#));
            assert_eq!(n, 7);
            let error = specs.err().unwrap_or_else(|| panic!("{edges} was taken"));
            assert!(error.contains(says), "{edges}: {error}");
        }
    }

    #[test]
    fn a_line_with_no_batch_to_answer_is_a_fault() {
        for line in [
            "nonsense",
            "[1]",
            r#"{"edges": []}"#,
            r#"{"batch": -1, "edges": []}"#,
            r#"{"stop": "e"}"#,
            r#"{"stop": [], "batch": 1}"#,
            r#"{"listen": {"host": "h", "keys": []}, "batch": 1}"#,
        ] {
            assert!(parse_command(line).is_err(), "{line} was taken");
        }
    }

    #[test]
    fn secrets_compare_whole() {
        assert!(same(b"s3cret", b"s3cret"));
        assert!(!same(b"s3cre", b"s3cret"));
        assert!(!same(b"s3cret!", b"s3cret"));
        assert!(!same(b"S3cret", b"s3cret"));
        assert!(!same(b"", b"s3cret"));
    }

    #[test]
    fn a_data_port_binds_the_family_its_host_is_written_in() {
        assert!(port_address("127.0.0.1").expect("IPv4").is_ipv4());
        assert!(port_address("::1").expect("IPv6").is_ipv6());
        assert!(port_address("[::1]").expect("IPv6").is_ipv6());
        assert!(port_address("::nonsense::").is_err());
    }
}
