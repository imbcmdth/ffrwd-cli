//! The relay (`ffrwd-wasm relay`), driven through its stdin the way a run
//! drives it. The processes on either end of an edge are this test, opening
//! the pipe paths as files, which is what ffmpeg and the sidecar do; a cut
//! edge's far side is a second relay, or a socket this test opens itself.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpStream};
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

const ROW: &str = "ffrwd:row ";
const SECRET_ENV: &str = "FFRWD_NODE_SECRET";

/// How long anything that should happen is given to.
const SOON: Duration = Duration::from_secs(10);

const KIB: usize = 1 << 10;
const MIB: usize = 1 << 20;

/// One running relay, the rows it has said, and the last flow row of each
/// edge.
struct Relay {
    child: Child,
    stdin: Option<ChildStdin>,
    rows: Receiver<Value>,
    flows: HashMap<String, Value>,
    batch: u64,
}

impl Relay {
    fn start() -> Relay {
        Relay::spawn(None)
    }

    fn with_secret(secret: &str) -> Relay {
        Relay::spawn(Some(secret))
    }

    fn spawn(secret: Option<&str>) -> Relay {
        let mut command = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"));
        command
            .arg("relay")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .env_remove(SECRET_ENV);
        if let Some(secret) = secret {
            command.env(SECRET_ENV, secret);
        }
        let mut child = command.spawn().expect("spawn the relay");
        let stdin = child.stdin.take();
        let rows = rows_of(child.stderr.take().expect("stderr was piped"));
        Relay {
            child,
            stdin,
            rows,
            flows: HashMap::new(),
            batch: 0,
        }
    }

    fn send(&mut self, command: &Value) {
        let stdin = self.stdin.as_mut().expect("stdin is open");
        writeln!(stdin, "{command}").expect("write a command");
        stdin.flush().expect("flush a command");
    }

    fn note(&mut self, row: &Value) {
        if row["kind"] == "flow" {
            let edge = row["edge"].as_str().expect("a flow row names its edge");
            self.flows.insert(edge.to_string(), row.clone());
        }
    }

    /// The next row `matches` takes. Flow rows on the way are remembered.
    fn until(&mut self, what: &str, matches: impl Fn(&Value) -> bool) -> Value {
        let deadline = Instant::now() + SOON;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.rows.recv_timeout(left) {
                Ok(row) => {
                    self.note(&row);
                    if matches(&row) {
                        return row;
                    }
                }
                Err(_) => panic!(
                    "the relay said nothing of {what} in {SOON:?}; its edges last said {:?}",
                    self.flows
                ),
            }
        }
    }

    /// The last flow row `edge` wrote, of the rows already said.
    fn flow(&mut self, edge: &str) -> Value {
        while let Ok(row) = self.rows.try_recv() {
            self.note(&row);
        }
        self.flows.get(edge).cloned().unwrap_or(Value::Null)
    }

    /// The flow row an edge ends with, whether it was said already or is
    /// still to come.
    fn done(&mut self, edge: &str) -> Value {
        let last = self.flow(edge);
        if last["done"] == true {
            return last;
        }
        self.until(&format!("{edge} ending"), |row| {
            row["kind"] == "flow" && row["edge"] == edge && row["done"] == true
        })
    }

    /// Send a batch and wait for the answer: None when it is ready, the
    /// error when it is not.
    fn try_open(&mut self, edges: Value) -> Option<String> {
        self.batch += 1;
        let batch = self.batch;
        self.send(&json!({"edges": edges, "batch": batch}));
        let answer = self.until(&format!("batch {batch}"), |row| {
            row["kind"] == "relay" && (row["ready"] == batch || row["batch"] == batch)
        });
        answer["error"].as_str().map(str::to_string)
    }

    fn open(&mut self, edges: Value) {
        if let Some(error) = self.try_open(edges) {
            panic!("the relay refused a good batch: {error}");
        }
    }

    fn listen(&mut self, keys: &[&str]) -> (String, u16) {
        self.send(&json!({"listen": {"host": "127.0.0.1", "keys": keys}}));
        let row = self.until("its data port", |row| {
            row["kind"] == "relay" && (row.get("listening").is_some() || row.get("error").is_some())
        });
        let listening = row["listening"]
            .as_array()
            .unwrap_or_else(|| panic!("no data port: {row}"));
        let host = listening[0].as_str().expect("a host").to_string();
        let port = listening[1].as_u64().expect("a port") as u16;
        (host, port)
    }

    /// Close stdin and wait for the relay to exit.
    fn close(&mut self) -> ExitStatus {
        drop(self.stdin.take());
        self.exit()
    }

    fn exit(&mut self) -> ExitStatus {
        let deadline = Instant::now() + SOON;
        loop {
            if let Some(status) = self.child.try_wait().expect("ask after the relay") {
                return status;
            }
            assert!(
                Instant::now() < deadline,
                "the relay did not exit in {SOON:?}"
            );
            thread::sleep(Duration::from_millis(20));
        }
    }
}

impl Drop for Relay {
    fn drop(&mut self) {
        // This test's own child, left running only by a failed assertion.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The rows a relay writes on stderr, as they arrive; its other lines are
/// passed on to this test's own stderr.
fn rows_of(stderr: impl Read + Send + 'static) -> Receiver<Value> {
    let (heard, rows) = mpsc::channel();
    thread::spawn(move || {
        for line in BufReader::new(stderr).lines() {
            let Ok(line) = line else { break };
            match line.strip_prefix(ROW) {
                Some(text) => {
                    let row: Value = serde_json::from_str(text)
                        .unwrap_or_else(|e| panic!("a row that is not JSON ({e}): {text}"));
                    if heard.send(row).is_err() {
                        break;
                    }
                }
                None => eprintln!("relay: {line}"),
            }
        }
    });
    rows
}

/// A pipe path of this test's own.
fn pipe(name: &str) -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let name = format!("ffrwd-relay-test-{}-{name}-{n}", std::process::id());
    if cfg!(windows) {
        format!(r"\\.\pipe\{name}")
    } else {
        std::env::temp_dir()
            .join(name)
            .to_str()
            .expect("a UTF-8 temporary directory")
            .to_string()
    }
}

/// An edge as the host writes one: every field, depth and buffer included.
fn edge(id: &str, from: impl Into<Value>, to: impl Into<Value>, depth: usize) -> Value {
    json!({"id": id, "from": from.into(), "to": to.into(), "depth": depth, "buffer": 65536, "spool": false})
}

fn producer(path: &str) -> File {
    OpenOptions::new()
        .write(true)
        .open(path)
        .unwrap_or_else(|e| panic!("open {path} to write: {e}"))
}

fn consumer(path: &str) -> File {
    File::open(path).unwrap_or_else(|e| panic!("open {path} to read: {e}"))
}

/// Bytes no two neighbouring stretches of which are alike.
fn pattern(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect()
}

/// `work` on a thread of its own, and a way to have its answer.
fn later<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> Receiver<T> {
    let (sent, answer) = mpsc::channel();
    thread::spawn(move || {
        let _ = sent.send(work());
    });
    answer
}

fn wait_for<T>(what: &str, answer: &Receiver<T>) -> T {
    match answer.recv_timeout(SOON) {
        Ok(value) => value,
        Err(RecvTimeoutError::Timeout) => panic!("{what} took longer than {SOON:?}"),
        Err(RecvTimeoutError::Disconnected) => panic!("{what} panicked"),
    }
}

fn write_all_of(path: String, bytes: Arc<Vec<u8>>, piece: usize) -> Receiver<()> {
    later(move || {
        let mut file = producer(&path);
        for part in bytes.chunks(piece) {
            file.write_all(part).expect("write the producer's bytes");
        }
    })
}

fn read_all_of(path: String) -> Receiver<Vec<u8>> {
    later(move || {
        let mut got = Vec::new();
        consumer(&path)
            .read_to_end(&mut got)
            .expect("read the consumer's bytes");
        got
    })
}

/// Write to `path` until a write fails: the error.
fn write_until_it_fails(path: String) -> Receiver<io::Error> {
    later(move || {
        let mut file = producer(&path);
        let piece = vec![7u8; 64 * KIB];
        loop {
            if let Err(e) = file.write_all(&piece) {
                return e;
            }
        }
    })
}

fn same_bytes(what: &str, got: &[u8], sent: &[u8]) {
    assert_eq!(got.len(), sent.len(), "{what}: the length");
    if let Some(at) = got.iter().zip(sent).position(|(a, b)| a != b) {
        panic!("{what}: the bytes differ from offset {at}");
    }
}

#[test]
fn every_byte_crosses_in_order_on_every_edge_of_a_batch() {
    let mut relay = Relay::start();
    let (a_in, a_out, b_in, b_out) = (pipe("a-in"), pipe("a-out"), pipe("b-in"), pipe("b-out"));
    // `a` asks for pipes bigger than the platform's default, which Linux
    // grows once they are open.
    let mut a = edge("a", a_in.as_str(), a_out.as_str(), MIB);
    a["buffer"] = json!(MIB);
    relay.open(json!([
        a,
        edge("b", b_in.as_str(), b_out.as_str(), 64 * KIB)
    ]));

    let a = Arc::new(pattern(8 * MIB, 1));
    let b = Arc::new(pattern(3 * MIB + 17, 2));
    let wrote_a = write_all_of(a_in, Arc::clone(&a), 100 * KIB);
    let wrote_b = write_all_of(b_in, Arc::clone(&b), 7 * KIB);
    let read_a = read_all_of(a_out);
    let read_b = read_all_of(b_out);
    wait_for("writing a", &wrote_a);
    wait_for("writing b", &wrote_b);
    same_bytes("edge a", &wait_for("reading a", &read_a), &a);
    same_bytes("edge b", &wait_for("reading b", &read_b), &b);

    for (id, sent) in [("a", &a), ("b", &b)] {
        let last = relay.done(id);
        assert_eq!(last["moved"], sent.len(), "{id}: {last}");
        assert_eq!(last["began"], true, "{id}: {last}");
        assert_eq!(last["opening"], false, "{id}: {last}");
        assert_eq!(last["writing"], false, "{id}: {last}");
    }
    assert!(relay.close().success());
}

#[test]
fn a_read_hands_back_what_has_arrived_rather_than_a_full_chunk() {
    let mut relay = Relay::start();
    let (from, to) = (pipe("partial-in"), pipe("partial-out"));
    relay.open(json!([edge(
        "partial",
        from.as_str(),
        to.as_str(),
        64 * KIB
    )]));
    let got = wait_for(
        "a short read",
        &later(move || {
            let mut file = producer(&from);
            file.write_all(&[b'c'; 300]).expect("write 300 bytes");
            let mut buf = vec![0u8; 64 * KIB];
            let n = consumer(&to).read(&mut buf).expect("read");
            drop(file);
            buf.truncate(n);
            buf
        }),
    );
    assert_eq!(got, vec![b'c'; 300]);
}

#[test]
fn a_deep_edge_reads_its_depth_ahead_of_a_consumer_not_yet_open_and_no_further() {
    let mut relay = Relay::start();
    let (from, to) = (pipe("deep-in"), pipe("deep-out"));
    let depth = MIB;
    relay.open(json!([edge("deep", from.as_str(), to.as_str(), depth)]));

    let sent = Arc::new(pattern(4 * MIB, 3));
    let written = Arc::new(AtomicU64::new(0));
    let producing = {
        let (sent, written) = (Arc::clone(&sent), Arc::clone(&written));
        later(move || {
            let mut file = producer(&from);
            for part in sent.chunks(64 * KIB) {
                file.write_all(part).expect("write the producer's bytes");
                written.fetch_add(part.len() as u64, Ordering::SeqCst);
            }
        })
    };

    // The producer gets the depth written and is then held: it stops moving.
    let deadline = Instant::now() + SOON;
    let mut last = 0;
    let mut still_since = Instant::now();
    loop {
        thread::sleep(Duration::from_millis(50));
        let now = written.load(Ordering::SeqCst);
        if now != last {
            last = now;
            still_since = Instant::now();
        } else if now > 0 && still_since.elapsed() > Duration::from_millis(750) {
            break;
        }
        assert!(Instant::now() < deadline, "the producer never stopped");
    }
    let held = last as usize;
    assert!(
        held >= depth,
        "the producer was held at {held}, short of the depth"
    );
    assert!(
        held <= depth + 512 * KIB,
        "the producer got {held} bytes in, well past a depth of {depth}"
    );
    let flow = relay.until("the edge opening", |row| {
        row["kind"] == "flow" && row["edge"] == "deep" && row["opening"] == true
    });
    assert_eq!(flow["moved"], 0, "nothing was handed on: {flow}");

    let got = wait_for("reading", &read_all_of(to));
    wait_for("writing", &producing);
    same_bytes("the deep edge", &got, &sent);
    assert_eq!(relay.done("deep")["moved"], sent.len());
}

#[test]
fn a_spool_never_holds_up_its_producer() {
    let mut relay = Relay::start();
    let (from, to) = (pipe("rows-in"), pipe("rows-out"));
    let mut rows = edge("rows", from.as_str(), to.as_str(), 64 * KIB);
    rows["spool"] = json!(true);
    relay.open(json!([rows]));

    // 7 MiB, past what a spool holds in memory, all of it written and the
    // pipe closed with no consumer there at all.
    let sent: Arc<Vec<u8>> = Arc::new(b"{\"n\": 1}\n".repeat(800_000));
    wait_for(
        "a producer with nobody reading",
        &write_all_of(from, Arc::clone(&sent), 64 * KIB),
    );
    let flow = relay.until("the spool counting what arrived", |row| {
        row["kind"] == "flow" && row["edge"] == "rows" && row["moved"] == sent.len()
    });
    assert_eq!(flow["began"], true, "{flow}");
    assert_eq!(flow["opening"], true, "{flow}");

    same_bytes("the spool", &wait_for("reading", &read_all_of(to)), &sent);
    assert_eq!(relay.done("rows")["moved"], sent.len());

    // What went to disk went with the edge.
    let spilled = format!("ffrwd-relay-{}-", relay.child.id());
    let left: Vec<_> = std::fs::read_dir(std::env::temp_dir())
        .expect("list the temporary directory")
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().starts_with(&spilled))
        .map(|entry| entry.path())
        .collect();
    assert!(left.is_empty(), "the spool left {left:?}");
}

#[test]
fn a_stop_breaks_off_an_end_waiting_in_a_read_or_a_write() {
    let mut relay = Relay::start();
    let paths: Vec<String> = ["idle-in", "idle-out", "stuck-in", "stuck-out"]
        .into_iter()
        .map(pipe)
        .collect();
    relay.open(json!([
        edge("idle", paths[0].as_str(), paths[1].as_str(), 64 * KIB),
        edge("stuck", paths[2].as_str(), paths[3].as_str(), 64 * KIB),
    ]));

    // `idle`: both ends open and quiet, so the relay waits in a read of the
    // producer's end. The producer writes again only after the stop.
    let (go, went) = mpsc::channel::<()>();
    let (idle_in, idle_out) = (paths[0].clone(), paths[1].clone());
    let producing = later(move || {
        let mut file = producer(&idle_in);
        file.write_all(b"hello").expect("write a little");
        let _ = went.recv();
        (0..100)
            .find_map(|_| file.write_all(&[0u8; 64 * KIB]).err())
            .is_some()
    });
    let consuming = later(move || {
        let mut file = consumer(&idle_out);
        let mut first = [0u8; 5];
        file.read_exact(&mut first).expect("read a little");
        let mut rest = Vec::new();
        let _ = file.read_to_end(&mut rest);
        (first, rest)
    });

    // `stuck`: a consumer that opens and never reads, so the relay waits in
    // a write once the pipe is full.
    let held_open = later({
        let stuck_out = paths[3].clone();
        move || consumer(&stuck_out)
    });
    let stuck_failed = write_until_it_fails(paths[2].clone());

    relay.until("idle moving", |row| {
        row["edge"] == "idle" && row["began"] == true
    });
    relay.until("stuck waiting on its consumer", |row| {
        row["edge"] == "stuck" && row["writing"] == true
    });
    thread::sleep(Duration::from_millis(300));
    relay.send(&json!({"stop": ["idle", "stuck"]}));

    let (first, rest) = wait_for("the idle consumer's read ending", &consuming);
    assert_eq!(&first, b"hello");
    assert!(rest.is_empty(), "{} bytes after a stop", rest.len());
    relay.done("idle");
    relay.done("stuck");
    let _ = go.send(());
    assert!(wait_for("the idle producer's write failing", &producing));
    wait_for("the stuck producer's write failing", &stuck_failed);
    drop(wait_for("the stuck consumer opening", &held_open));
    assert!(relay.close().success());
}

#[test]
fn a_consumer_that_never_opens_leaves_the_edge_opening_until_it_is_stopped() {
    let mut relay = Relay::start();
    let (from, to) = (pipe("abandoned-in"), pipe("abandoned-out"));
    relay.open(json!([edge(
        "abandoned",
        from.as_str(),
        to.as_str(),
        64 * KIB
    )]));
    let failed = write_until_it_fails(from);
    relay.until("the edge opening", |row| {
        row["kind"] == "flow" && row["edge"] == "abandoned" && row["opening"] == true
    });
    thread::sleep(Duration::from_millis(600));
    let flow = relay.flow("abandoned");
    assert_eq!(flow["opening"], true, "{flow}");
    assert_eq!(flow["done"], false, "{flow}");

    relay.send(&json!({"stop": ["abandoned"]}));
    let last = relay.done("abandoned");
    assert_eq!(last["opening"], false, "{last}");
    wait_for("the producer's write failing", &failed);
    if cfg!(unix) {
        assert!(!std::path::Path::new(&to).exists(), "{to} was left behind");
    }
    assert!(relay.close().success());
}

#[test]
fn a_consumer_that_leaves_early_breaks_the_producers_write() {
    let mut relay = Relay::start();
    let (from, to) = (pipe("early-in"), pipe("early-out"));
    relay.open(json!([edge("early", from.as_str(), to.as_str(), 64 * KIB)]));
    let failed = write_until_it_fails(from);
    wait_for(
        "the consumer's first bytes",
        &later(move || {
            let mut buf = vec![0u8; 64 * KIB];
            consumer(&to).read_exact(&mut buf).expect("read a little");
        }),
    );
    let error = wait_for("the producer's write failing", &failed);
    eprintln!("the producer was told: {error}");
    let last = relay.done("early");
    assert_eq!(last["began"], true, "{last}");
}

#[test]
fn closing_stdin_ends_every_edge_and_the_relay() {
    let mut relay = Relay::start();
    let paths: Vec<String> = ["quiet-in", "quiet-out", "busy-in", "busy-out"]
        .into_iter()
        .map(pipe)
        .collect();
    relay.open(json!([
        edge("quiet", paths[0].as_str(), paths[1].as_str(), 64 * KIB),
        edge("busy", paths[2].as_str(), paths[3].as_str(), 64 * KIB),
    ]));
    let failed = write_until_it_fails(paths[2].clone());
    relay.until("the busy edge opening", |row| {
        row["kind"] == "flow" && row["edge"] == "busy" && row["opening"] == true
    });

    assert!(relay.close().success());
    relay.done("quiet");
    relay.done("busy");
    wait_for("the producer's write failing", &failed);
    for path in &paths {
        if cfg!(unix) {
            assert!(
                !std::path::Path::new(path).exists(),
                "{path} was left behind"
            );
        } else {
            assert!(File::open(path).is_err(), "{path} is still there");
        }
    }
}

#[test]
fn what_the_relay_cannot_take_is_refused() {
    // An argv: the relay's edges arrive on stdin.
    let ran = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .args(["relay", "-edge", "e"])
        .stdin(Stdio::null())
        .output()
        .expect("run the relay");
    assert_eq!(ran.status.code(), Some(2));
    let said = String::from_utf8_lossy(&ran.stderr);
    assert!(
        said.contains(r#"ffrwd:row {"kind":"relay","error":"relay takes no arguments"#),
        "{said}"
    );

    // A batch it cannot set up is that batch's error, and the relay carries on.
    let mut relay = Relay::start();
    let error = relay
        .try_open(json!([{"id": "e", "from": "x"}]))
        .expect("a batch with no destination");
    assert!(error.contains("missing field `to`"), "{error}");
    let error = relay
        .try_open(json!([edge("e", "x", "y", 0)]))
        .expect("a depth of nothing");
    assert!(error.contains("depth of 0"), "{error}");

    let (from, to) = (pipe("after-in"), pipe("after-out"));
    relay.open(json!([edge("after", from.as_str(), to.as_str(), 64 * KIB)]));
    let error = relay
        .try_open(json!([edge(
            "after",
            pipe("x").as_str(),
            pipe("y").as_str(),
            64 * KIB
        )]))
        .expect("an id already running");
    assert!(error.contains("already running"), "{error}");
    let error = relay
        .try_open(json!([edge(
            "again",
            from.as_str(),
            pipe("z").as_str(),
            64 * KIB
        )]))
        .expect("a pipe already made");
    assert!(error.contains("making"), "{error}");

    let wrote = write_all_of(from, Arc::new(b"f".repeat(10)), 10);
    assert_eq!(wait_for("reading", &read_all_of(to)), b"f".repeat(10));
    wait_for("writing", &wrote);
    relay.done("after");

    // A line it cannot answer at all is the relay's fault.
    relay.send(&json!("nonsense"));
    let fault = relay.until("the fault", |row| {
        row["kind"] == "relay" && row.get("error").is_some()
    });
    assert!(fault.get("batch").is_none(), "{fault}");
    assert_eq!(relay.exit().code(), Some(1));
}

#[test]
fn a_cut_edge_crosses_from_one_relay_to_another_byte_exact() {
    let mut listening = Relay::with_secret("s3cret");
    let mut dialing = Relay::with_secret("s3cret");
    let (host, port) = listening.listen(&["k1"]);
    let (from, to) = (pipe("cut-in"), pipe("cut-out"));

    dialing.open(json!([edge(
        "out",
        from.as_str(),
        json!({"dial": [host, port], "key": "k1"}),
        MIB
    )]));
    listening.open(json!([edge(
        "in",
        json!({"listen": "k1"}),
        to.as_str(),
        MIB
    )]));

    let sent = Arc::new(pattern(6 * MIB + 5, 4));
    let wrote = write_all_of(from, Arc::clone(&sent), 64 * KIB);
    let got = wait_for("reading across the cut", &read_all_of(to));
    wait_for("writing into the cut", &wrote);
    same_bytes("the cut edge", &got, &sent);
    assert_eq!(dialing.done("out")["moved"], sent.len());
    assert_eq!(listening.done("in")["moved"], sent.len());
    assert!(dialing.close().success());
    assert!(listening.close().success());
}

#[test]
fn a_dial_the_listener_refuses_fails_its_edge() {
    let mut listening = Relay::with_secret("s3cret");
    let mut dialing = Relay::with_secret("not-it");
    let (host, port) = listening.listen(&["k1"]);
    let from = pipe("refused-in");
    dialing.open(json!([edge(
        "out",
        from.as_str(),
        json!({"dial": [host, port], "key": "k1"}),
        64 * KIB
    )]));
    let failed = dialing.until("the edge failing", |row| {
        row["kind"] == "relay" && row["edge"] == "out"
    });
    assert!(
        failed["error"].as_str().is_some_and(|e| e.contains("k1")),
        "{failed}"
    );
    dialing.done("out");
    let refused = listening.until("the refusal", |row| row.get("refused").is_some());
    assert_eq!(refused["refused"], "the wrong secret");
}

#[test]
fn the_data_port_refuses_what_is_not_a_cut_edge_of_this_job() {
    let mut relay = Relay::with_secret("s3cret");
    let (host, port) = relay.listen(&["k1"]);
    let connect = || {
        let socket = TcpStream::connect((host.as_str(), port)).expect("reach the data port");
        socket
            .set_read_timeout(Some(SOON))
            .expect("bound the test's reads");
        socket
    };
    let answer = |socket: &mut TcpStream| {
        let mut got = Vec::new();
        let _ = socket.read_to_end(&mut got);
        got
    };

    let mut taken = connect();
    taken
        .write_all(b"FFRWD-CUT 1 s3cret k1\n")
        .expect("send an opening line");
    let mut ok = [0u8; 3];
    taken.read_exact(&mut ok).expect("hear the answer");
    assert_eq!(&ok, b"OK\n");

    let refusals: [(&[u8], &str); 6] = [
        (b"FFRWD-CUT 1 nope k1\n", "the wrong secret"),
        (
            b"FFRWD-CUT 1 s3cret k9\n",
            "an edge this node does not listen for: k9",
        ),
        (b"FFRWD-CUT 1 s3cret k1\n", "a second connection for k1"),
        (
            b"\x8bNUT\x00\x02main stream\n",
            "not a cut edge's opening line",
        ),
        (b"FFRWD-CUT 2 s3cret k1\n", "not a cut edge's opening line"),
        (&[b'x'; 300], "an opening line too long"),
    ];
    for (line, reason) in refusals {
        let mut socket = connect();
        socket.write_all(line).expect("send an opening line");
        let row = relay.until(reason, |row| row.get("refused").is_some());
        assert_eq!(row["refused"], reason);
        assert_eq!(row["kind"], "relay");
        assert!(answer(&mut socket).is_empty(), "{reason}: answered");
    }

    let mut socket = connect();
    socket
        .shutdown(Shutdown::Write)
        .expect("close without a word");
    let row = relay.until("a silent close", |row| row.get("refused").is_some());
    assert_eq!(row["refused"], "closed before its opening line");
    assert!(answer(&mut socket).is_empty());
}

#[test]
fn a_data_port_needs_the_jobs_secret() {
    let mut relay = Relay::start();
    relay.send(&json!({"listen": {"host": "127.0.0.1", "keys": ["k1"]}}));
    let row = relay.until("the listen failing", |row| row["kind"] == "relay");
    assert!(
        row["error"]
            .as_str()
            .is_some_and(|e| e.contains(SECRET_ENV)),
        "{row}"
    );
    // And the relay carries on.
    let (from, to) = (pipe("still-in"), pipe("still-out"));
    relay.open(json!([edge("still", from.as_str(), to.as_str(), 64 * KIB)]));
    assert!(relay.close().success());
}
