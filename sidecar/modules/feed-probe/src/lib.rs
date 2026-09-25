//! A feeder, read and reported: the module's own stream passes through
//! untouched, and a second stream, the feeder, arrives as NUT on a loopback
//! port the module listens on itself. Each picture the feeder sends is one
//! row, its pts in the feeder's own time base and its size.
//!
//! The port is the `port` param, which the host fills with a port it picked
//! when a stream is written in the feeder's place. A connection that ends
//! before it has said anything is dropped and the next one is taken, since
//! the host tries the port once before it starts the feeder's writer.
//!
//! The last call waits for the feeder to finish, up to `DRAIN_SECONDS`, so
//! every picture it sent is counted however the two streams raced. A run
//! with no feeder at all pays that wait once, at the end.

wit_bindgen::generate!({
    path: "../../wit",
    world: "window-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::window_filter::{
    Feeder, Format, FramePayload, Guest, InWindow, Meta, OutFrame, Processed, StreamInfo,
    WindowMeta,
};
use ffrwd_nut::{Event, Limits, PushDemuxer};
use serde::{Deserialize, Serialize};
use wasip2::clocks::monotonic_clock;
use wasip2::io::poll::poll;
use wasip2::io::streams::{InputStream, OutputStream, StreamError};
use wasip2::sockets::instance_network::instance_network;
use wasip2::sockets::network::{
    ErrorCode, IpAddressFamily, IpSocketAddress, Ipv4SocketAddress, Network,
};
use wasip2::sockets::tcp::TcpSocket;
use wasip2::sockets::tcp_create_socket::create_tcp_socket;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"port":{"type":"integer"}},"required":["port"],"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"feed_pts":{"type":"integer"},"w":{"type":"integer"},"h":{"type":"integer"}},"additionalProperties":false}"#;

/// How much one read asks for.
const READ_CHUNK: u64 = 256 << 10;
/// How long the last call waits for the feeder to finish.
const DRAIN_SECONDS: u64 = 20;
/// How long one wait inside the drain lasts before the clock is read again.
const WAIT_NANOS: u64 = 100_000_000;

#[derive(Deserialize)]
struct Params {
    port: u16,
}

#[derive(Serialize)]
struct Row {
    feed_pts: i64,
    w: u32,
    h: u32,
}

/// One accepted connection and what has been read off it.
struct Conn {
    demux: PushDemuxer,
    /// The picture's stream index and size, once its header has arrived.
    video: Option<(usize, u32, u32)>,
    /// Whether anything at all has arrived.
    heard: bool,
    closed: bool,
    input: InputStream,
    _output: OutputStream,
    /// Named after its streams so it outlives them.
    _socket: TcpSocket,
}

impl Conn {
    /// Whatever has arrived, parsed, each picture a row in `rows`.
    fn read(&mut self, rows: &mut Vec<String>) {
        loop {
            match self.input.read(READ_CHUNK) {
                Ok(bytes) if bytes.is_empty() => break,
                Ok(bytes) => {
                    self.heard = true;
                    self.demux.feed(&bytes);
                }
                Err(StreamError::Closed) | Err(StreamError::LastOperationFailed(_)) => {
                    self.closed = true;
                    self.demux.finish();
                    break;
                }
            }
        }
        loop {
            match self.demux.next_event() {
                Ok(Some(Event::StreamHeader { index, stream })) => {
                    if self.video.is_none() {
                        if let Some((w, h)) = stream.video_geometry() {
                            self.video = Some((index, w, h));
                        }
                    }
                }
                Ok(Some(Event::Frame { stream, packet })) => {
                    if let Some((index, w, h)) = self.video {
                        if stream == index {
                            let row = Row {
                                feed_pts: packet.pts,
                                w,
                                h,
                            };
                            rows.push(serde_json::to_string(&row).expect("row serializes"));
                        }
                    }
                }
                Ok(Some(Event::EndOfInput)) | Ok(None) => break,
                Ok(Some(_)) => {}
                Err(err) => {
                    eprintln!("feed-probe: the feeder's stream was refused: {err}");
                    self.closed = true;
                    break;
                }
            }
        }
    }
}

struct State {
    conn: Option<Conn>,
    /// Whether a feeder has come and gone: nothing more is taken after it.
    done: bool,
    listener: TcpSocket,
    /// Named last so it outlives the socket made from it.
    _network: Network,
}

impl State {
    fn open(port: u16) -> Result<State, String> {
        let network = instance_network();
        let listener = create_tcp_socket(IpAddressFamily::Ipv4)
            .map_err(|code| format!("feed-probe: no socket: {}", code.name()))?;
        let at = IpSocketAddress::Ipv4(Ipv4SocketAddress {
            port,
            address: (127, 0, 0, 1),
        });
        listener
            .start_bind(&network, at)
            .and_then(|()| settle(|| listener.finish_bind()))
            .map_err(|code| {
                format!(
                    "feed-probe: 127.0.0.1:{port} cannot be bound: {}",
                    code.name()
                )
            })?;
        listener
            .start_listen()
            .and_then(|()| settle(|| listener.finish_listen()))
            .map_err(|code| {
                format!(
                    "feed-probe: 127.0.0.1:{port} cannot be listened on: {}",
                    code.name()
                )
            })?;
        Ok(State {
            conn: None,
            done: false,
            listener,
            _network: network,
        })
    }

    /// Takes a waiting connection, reads what has arrived, and says whether
    /// the feeder has finished. Never waits.
    fn step(&mut self, rows: &mut Vec<String>) -> bool {
        if self.done {
            return true;
        }
        if self.conn.is_none() {
            if let Ok((socket, input, output)) = self.listener.accept() {
                self.conn = Some(Conn {
                    demux: PushDemuxer::new(Limits::default()),
                    video: None,
                    heard: false,
                    closed: false,
                    input,
                    _output: output,
                    _socket: socket,
                });
            }
        }
        let Some(conn) = self.conn.as_mut() else {
            return false;
        };
        conn.read(rows);
        if !conn.closed {
            return false;
        }
        // One that said nothing was somebody trying the port.
        self.done = conn.heard;
        self.conn = None;
        self.done
    }

    /// Steps until the feeder has finished or the wait runs out.
    fn drain(&mut self, rows: &mut Vec<String>) {
        let until = monotonic_clock::now() + DRAIN_SECONDS * 1_000_000_000;
        while !self.step(rows) && monotonic_clock::now() < until {
            let timer = monotonic_clock::subscribe_duration(WAIT_NANOS);
            match self.conn.as_ref() {
                Some(conn) => {
                    let ready = conn.input.subscribe();
                    poll(&[&ready, &timer]);
                }
                None => {
                    let ready = self.listener.subscribe();
                    poll(&[&ready, &timer]);
                }
            }
        }
    }
}

/// Drives a `finish-*` call past the would-block it may answer first.
fn settle(mut step: impl FnMut() -> Result<(), ErrorCode>) -> Result<(), ErrorCode> {
    loop {
        match step() {
            Err(ErrorCode::WouldBlock) => continue,
            other => return other,
        }
    }
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

fn params(text: &str) -> Result<Params, String> {
    serde_json::from_str(text).map_err(|err| format!("feed-probe: bad params: {err}"))
}

struct FeedProbe;

impl Guest for FeedProbe {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "feed-probe".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
                pixel_formats: vec!["yuv420p".to_string()],
                sample_formats: vec![],
                sample_rates: vec![],
                channel_counts: vec![],
                rows_language: vec![],
            },
            window: 1,
            stride: 1,
            // The connection is state carried from call to call.
            pure: false,
            one_to_one: true,
            reads_rows: false,
            forwards_rows: false,
            inputs: 1,
            feeders: vec![Feeder {
                input: 1,
                port_param: "port".to_string(),
                kind: "video".to_string(),
                group: String::new(),
            }],
        }
    }

    fn init(_format: Format, _stream_info: StreamInfo, params_text: String) -> Result<(), String> {
        let wanted = params(&params_text)?;
        let state = State::open(wanted.port)?;
        STATE.with(|cell| *cell.borrow_mut() = Some(state));
        Ok(())
    }

    fn set_params(params_text: String) -> Result<(), String> {
        params(&params_text).map(|_| ())
    }

    fn process(window: &InWindow, _trailing: Vec<String>, last: bool) -> Processed {
        let mut rows = Vec::new();
        STATE.with(|cell| {
            if let Some(state) = cell.borrow_mut().as_mut() {
                if last {
                    state.drain(&mut rows);
                } else {
                    state.step(&mut rows);
                }
            }
        });
        let frames: Vec<OutFrame> = (0..window.len())
            .map(|i| OutFrame {
                pts: window.pts(i),
                frame: FramePayload::Same,
                rows: if i == 0 {
                    std::mem::take(&mut rows)
                } else {
                    vec![]
                },
            })
            .collect();
        Processed {
            frames,
            // Rows with no frame of this call to ride.
            trailing: rows,
        }
    }
}

export!(FeedProbe);
