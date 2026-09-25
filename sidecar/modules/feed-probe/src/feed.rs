//! A feeder connection, read and reported: the module listens on 127.0.0.1
//! at its `port` param, takes the connection the host's writer makes, and
//! turns each packet of the first stream it counts into a row. Shared by the
//! picture probe and the sound probe, which differ in what they count and
//! what a row says.
//!
//! A connection that ends before it has said anything is dropped and the next
//! one is taken, since the host tries the port once before it starts the
//! feeder's writer. The last call waits for the feeder to finish, up to
//! `DRAIN_SECONDS`, so every packet it sent is counted however the two
//! streams raced. A run with no feeder at all pays that wait once, at the end.

use std::marker::PhantomData;

use crate::exports::ffrwd::av::window_filter::{FramePayload, InWindow, OutFrame, Processed};
use ffrwd_nut::{Event, Limits, PushDemuxer, Stream};
use serde::Deserialize;
use wasip2::clocks::monotonic_clock;
use wasip2::io::poll::poll;
use wasip2::io::streams::{InputStream, OutputStream, StreamError};
use wasip2::sockets::instance_network::instance_network;
use wasip2::sockets::network::{
    ErrorCode, IpAddressFamily, IpSocketAddress, Ipv4SocketAddress, Network,
};
use wasip2::sockets::tcp::TcpSocket;
use wasip2::sockets::tcp_create_socket::create_tcp_socket;

pub const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"port":{"type":"integer"}},"required":["port"],"additionalProperties":false}"#;

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

/// What one probe counts off its feeder.
pub trait Probe {
    /// The module's name, for its messages.
    const NAME: &'static str;
    /// Whether this is the stream it counts.
    fn counts(stream: &Stream) -> bool;
    /// The row one packet of that stream is.
    fn row(stream: &Stream, pts: i64, payload: &[u8]) -> String;
}

/// One accepted connection and what has been read off it.
struct Conn {
    demux: PushDemuxer,
    /// The counted stream's index and header, once it has arrived.
    counted: Option<(usize, Stream)>,
    /// Whether anything at all has arrived.
    heard: bool,
    closed: bool,
    input: InputStream,
    _output: OutputStream,
    /// Named after its streams so it outlives them.
    _socket: TcpSocket,
}

impl Conn {
    /// Whatever has arrived, parsed, each counted packet a row in `rows`.
    fn read<P: Probe>(&mut self, rows: &mut Vec<String>) {
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
                    if self.counted.is_none() && P::counts(&stream) {
                        self.counted = Some((index, stream));
                    }
                }
                Ok(Some(Event::Frame { stream, packet })) => {
                    if let Some((index, header)) = &self.counted {
                        if stream == *index {
                            rows.push(P::row(header, packet.pts, self.demux.payload()));
                        }
                    }
                }
                Ok(Some(Event::EndOfInput)) | Ok(None) => break,
                Ok(Some(_)) => {}
                Err(err) => {
                    eprintln!("{}: the feeder's stream was refused: {err}", P::NAME);
                    self.closed = true;
                    break;
                }
            }
        }
    }
}

/// The listening socket, and the feeder connection once it has come.
pub struct Feed<P: Probe> {
    conn: Option<Conn>,
    /// Whether a feeder has come and gone: nothing more is taken after it.
    done: bool,
    listener: TcpSocket,
    /// Named last so it outlives the socket made from it.
    _network: Network,
    _probe: PhantomData<P>,
}

impl<P: Probe> Feed<P> {
    /// Listens on 127.0.0.1 at the port `params` names.
    pub fn open(params_text: &str) -> Result<Feed<P>, String> {
        let port = params(params_text, P::NAME)?.port;
        let network = instance_network();
        let listener = create_tcp_socket(IpAddressFamily::Ipv4)
            .map_err(|code| format!("{}: no socket: {}", P::NAME, code.name()))?;
        let at = IpSocketAddress::Ipv4(Ipv4SocketAddress {
            port,
            address: (127, 0, 0, 1),
        });
        listener
            .start_bind(&network, at)
            .and_then(|()| settle(|| listener.finish_bind()))
            .map_err(|code| {
                format!(
                    "{}: 127.0.0.1:{port} cannot be bound: {}",
                    P::NAME,
                    code.name()
                )
            })?;
        listener
            .start_listen()
            .and_then(|()| settle(|| listener.finish_listen()))
            .map_err(|code| {
                format!(
                    "{}: 127.0.0.1:{port} cannot be listened on: {}",
                    P::NAME,
                    code.name()
                )
            })?;
        Ok(Feed {
            conn: None,
            done: false,
            listener,
            _network: network,
            _probe: PhantomData,
        })
    }

    /// Takes a waiting connection, reads what has arrived, and says whether
    /// the feeder has finished. Never waits.
    pub fn step(&mut self, rows: &mut Vec<String>) -> bool {
        if self.done {
            return true;
        }
        if self.conn.is_none() {
            if let Ok((socket, input, output)) = self.listener.accept() {
                self.conn = Some(Conn {
                    demux: PushDemuxer::new(Limits::default()),
                    counted: None,
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
        conn.read::<P>(rows);
        if !conn.closed {
            return false;
        }
        // One that said nothing was somebody trying the port.
        self.done = conn.heard;
        self.conn = None;
        self.done
    }

    /// Steps until the feeder has finished or the wait runs out.
    pub fn drain(&mut self, rows: &mut Vec<String>) {
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

/// The params, checked; `name` is the module's, for the message.
pub fn check_params(text: &str, name: &str) -> Result<(), String> {
    params(text, name).map(|_| ())
}

fn params(text: &str, name: &str) -> Result<Params, String> {
    serde_json::from_str(text).map_err(|err| format!("{name}: bad params: {err}"))
}

/// One call: the module's own stream passes through untouched, and what the
/// feeder has sent since the last call rides its first payload as rows. The
/// last call waits for the feeder to finish first.
pub fn process<P: Probe>(feed: Option<&mut Feed<P>>, window: &InWindow, last: bool) -> Processed {
    let mut rows = Vec::new();
    if let Some(feed) = feed {
        if last {
            feed.drain(&mut rows);
        } else {
            feed.step(&mut rows);
        }
    }
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
        // Rows with no payload of this call to ride.
        trailing: rows,
    }
}
