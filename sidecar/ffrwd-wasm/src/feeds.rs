//! Port feeds: the host-owned loopback listener that serves a hold input
//! given by a port. Whatever connects and writes a NUT of raw video and PCM
//! is the input's source for as long as it stays, conformed to the port's
//! format on the way in: its picture to the size and pixel format the port
//! takes, its sound to the port's sample format. One connection at a time;
//! a second while one is on is accepted and closed at once, unless the one
//! on has sent nothing yet, in which case the newcomer takes its place. The
//! connection is read only while the input has room, so a source that
//! outruns the clock waits on its socket, paced by TCP.

use std::io::{self, Read};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use ffrwd_frame::yuv::{self, Colour};
use ffrwd_frame::Rgba;
use ffrwd_wasm::nut::{self, Event, Limits, Media, PushDemuxer};
use ffrwd_wasm_runtime::node::{StreamFormat, TickFrame};
use ffrwd_wasm_runtime::runtime::{AudioFormat, ColorInfo, Message, StreamInfo, VideoFormat};
use serde_json::json;

use crate::hold::SourceInfo;
use crate::lanes::{Room, Scheduler};
use crate::leaky::ROW_PREFIX;
use crate::node_graph::Carried;
use crate::tick::Item;
use std::sync::atomic::Ordering;

/// The receive buffer every connection asks for: about a dozen 1080p
/// frames, so a source has somewhere to write between two reads.
const RECEIVE_BUFFER: usize = 16 << 20;

/// How much one read takes off the socket.
const READ_CHUNK: usize = 256 << 10;

/// The largest frame a source may send before its header has said what it
/// is, and the largest run of samples after.
const MAX_FRAME: u64 = 64 << 20;
const MAX_AUDIO_PACKET: u64 = 192_000 * 8 * 4;

/// How long a read, or a wait for room, blocks before the listener looks
/// for another connection.
const POLL: Duration = Duration::from_millis(50);

/// One hold input a feed serves.
pub struct FeedMember {
    pub id: u32,
    pub name: String,
    /// What the port takes, which the source is conformed to.
    pub format: StreamFormat,
    /// The port reads its frames' times alone: nothing is conformed or
    /// carried for it.
    pub timing: bool,
}

/// One listener: a port, and the inputs its connections feed. The picture
/// comes first where there is one.
pub struct FeedSpec {
    pub node: String,
    pub port: u16,
    pub members: Vec<FeedMember>,
}

impl FeedSpec {
    fn names(&self) -> String {
        let names: Vec<&str> = self.members.iter().map(|m| m.name.as_str()).collect();
        format!("{} '{}'", self.node, names.join("', '"))
    }
}

/// The picture's conversion: one of the wire's two raw formats to the
/// port's, at the port's size.
struct Picture {
    width: u32,
    height: u32,
    pix_fmt: &'static str,
    /// The matrix each yuv side is converted in; an rgba side has none.
    colour: Option<Colour>,
    to: VideoFormat,
    to_colour: Option<Colour>,
}

impl Picture {
    fn same(&self) -> bool {
        self.width == self.to.width
            && self.height == self.to.height
            && self.pix_fmt == self.to.pix_fmt
    }

    fn convert(&self, data: &[u8]) -> Result<Vec<u8>> {
        if self.same() {
            return Ok(data.to_vec());
        }
        let (w, h) = (self.width as usize, self.height as usize);
        let rgba: Vec<u8> = match self.pix_fmt {
            "rgba" => data.to_vec(),
            _ => {
                let mut rgba = vec![0u8; w * h * 4];
                let frame = yuv::Yuv420p::new(data, w, h).map_err(|e| anyhow!(e))?;
                let colour = self
                    .colour
                    .ok_or_else(|| anyhow!("a yuv picture has no matrix"))?;
                yuv::to_rgba(&frame, colour, &mut rgba).map_err(|e| anyhow!(e))?;
                rgba
            }
        };
        let (tw, th) = (self.to.width as usize, self.to.height as usize);
        let rgba = if (tw, th) == (w, h) {
            rgba
        } else {
            resize_rgba(&rgba, w, h, tw, th)?
        };
        match self.to.pix_fmt {
            "rgba" => Ok(rgba),
            _ => {
                let mut out = vec![0u8; yuv::Yuv420p::size(tw, th)];
                let frame = Rgba::new(&rgba, tw, th).map_err(|e| anyhow!(e))?;
                let colour = self
                    .to_colour
                    .ok_or_else(|| anyhow!("a yuv picture has no matrix"))?;
                yuv::from_rgba(&frame, colour, &mut out).map_err(|e| anyhow!(e))?;
                Ok(out)
            }
        }
    }
}

/// `rgba` of `width` x `height` resized to `to_width` x `to_height`,
/// bilinear.
fn resize_rgba(
    rgba: &[u8],
    width: usize,
    height: usize,
    to_width: usize,
    to_height: usize,
) -> Result<Vec<u8>> {
    use fast_image_resize::images::{Image, ImageRef};
    use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
    let source = ImageRef::new(width as u32, height as u32, rgba, PixelType::U8x4)
        .map_err(|e| anyhow!("resizing a feed's picture: {e}"))?;
    let mut target = Image::new(to_width as u32, to_height as u32, PixelType::U8x4);
    let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear));
    Resizer::new()
        .resize(&source, &mut target, &options)
        .map_err(|e| anyhow!("resizing a feed's picture: {e}"))?;
    Ok(target.into_vec())
}

/// The sound's conversion: between the wire's two PCM formats.
struct Sound {
    sample_fmt: &'static str,
    to: AudioFormat,
}

impl Sound {
    fn convert(&self, data: &[u8]) -> Vec<u8> {
        match (self.sample_fmt, self.to.sample_fmt) {
            ("s16", "f32") => data
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|pair| (i16::from_le_bytes(*pair) as f32 / 32_768.0).to_le_bytes())
                .collect(),
            ("f32", "s16") => data
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|quad| {
                    let value = f32::from_le_bytes(*quad);
                    ((value * 32_768.0).round().clamp(-32_768.0, 32_767.0) as i16).to_le_bytes()
                })
                .collect(),
            _ => data.to_vec(),
        }
    }
}

enum Convert {
    Picture(Picture),
    Sound(Sound),
    /// A timing member's frames: their times, and a sound's length in
    /// samples of `width` bytes.
    Timing {
        audio: Option<usize>,
    },
    /// A data member's messages, as they are.
    Data,
}

/// One stream of a connection: which member it feeds, and how.
struct Lane {
    member: usize,
    base: nut::TimeBase,
    convert: Convert,
}

struct Conn {
    serial: u64,
    socket: TcpStream,
    demux: PushDemuxer,
    bytes: u64,
    /// Per stream of the NUT, the member it feeds.
    lanes: Vec<Option<Lane>>,
    opened: bool,
}

impl Conn {
    fn new(serial: u64, socket: TcpStream, streams: usize) -> Result<Conn> {
        socket.set_read_timeout(Some(POLL))?;
        Ok(Conn {
            serial,
            socket,
            demux: PushDemuxer::new(Limits {
                max_frame: MAX_FRAME,
                max_header: 64 << 10,
                max_buffered: 1 << 20,
                max_streams: streams.max(4),
            }),
            bytes: 0,
            lanes: Vec::new(),
            opened: false,
        })
    }
}

fn frame_colour(media: &nut::Media) -> Result<Colour> {
    let info = crate::color_from(media);
    Colour::of(info.as_ref().map(|c| yuv::ColorInfo {
        range: c.range,
        primaries: c.primaries,
        trc: c.trc,
        space: c.space,
    }))
    .map_err(|e| anyhow!(e))
}

fn port_colour(color: Option<&ColorInfo>) -> Result<Colour> {
    Colour::of(color.map(|c| yuv::ColorInfo {
        range: c.range,
        primaries: c.primaries,
        trc: c.trc,
        space: c.space,
    }))
    .map_err(|e| anyhow!(e))
}

/// Which wire streams feed which members, once the headers are whole: the
/// first picture to the picture member and the first sound to the sound
/// member. A connection may carry any of the members and more besides; a
/// member it does not carry is absent from its feed. A picture the port
/// cannot take refuses the connection; a sound it cannot take is left on
/// the wire and said.
fn match_streams(spec: &FeedSpec, conn: &mut Conn, report: &mut dyn FnMut(String)) -> Result<()> {
    let streams: Vec<Option<nut::Stream>> = conn.demux.streams().to_vec();
    conn.lanes = (0..streams.len()).map(|_| None).collect();
    let mut taken = vec![false; spec.members.len()];
    for (index, stream) in streams.iter().enumerate() {
        let Some(stream) = stream else { continue };
        let kind = match stream.media {
            Media::Video { .. } => "video",
            Media::Audio { .. } => "audio",
            Media::Other { .. } if stream.is_json() => "data",
            Media::Other { .. } => continue,
        };
        let member = spec.members.iter().enumerate().find_map(|(m, member)| {
            let wanted = match &member.format {
                StreamFormat::Video(_) => "video",
                StreamFormat::Audio(_) => "audio",
                StreamFormat::Data(_) => "data",
                _ => "",
            };
            (!taken[m] && wanted == kind).then_some(m)
        });
        let Some(member) = member else { continue };
        if spec.members[member].timing {
            let audio = match stream.media {
                Media::Audio { channels, .. } => {
                    let bytes = stream
                        .sample_fmt()
                        .map_or(0, |f| if f == "s16" { 2 } else { 4 });
                    if bytes == 0 {
                        continue;
                    }
                    conn.demux.set_stream_limit(index, MAX_AUDIO_PACKET);
                    Some(bytes * channels as usize)
                }
                _ => {
                    if let Some((width, height)) = stream.video_geometry() {
                        if let Some(pix_fmt) = stream.pix_fmt() {
                            let incoming = crate::frame_len_for(pix_fmt, width, height)?;
                            conn.demux.set_stream_limit(index, incoming as u64);
                        }
                    }
                    None
                }
            };
            taken[member] = true;
            conn.lanes[index] = Some(Lane {
                member,
                base: stream.time_base,
                convert: Convert::Timing { audio },
            });
            continue;
        }
        if kind == "data" {
            taken[member] = true;
            conn.lanes[index] = Some(Lane {
                member,
                base: stream.time_base,
                convert: Convert::Data,
            });
            continue;
        }
        let convert = match &spec.members[member].format {
            StreamFormat::Video(to) => {
                let Some((width, height)) = stream.video_geometry() else {
                    anyhow::bail!("the feeder's picture carries no frame size");
                };
                let pix_fmt = stream.pix_fmt().ok_or_else(|| {
                    anyhow!(
                        "the feeder's picture is {}, and a feed is rgba or yuv420p",
                        stream.fourcc_name()
                    )
                })?;
                if !matches!(pix_fmt, "rgba" | "yuv420p")
                    || !matches!(to.pix_fmt, "rgba" | "yuv420p")
                {
                    anyhow::bail!(
                        "the feeder's picture is {pix_fmt} and this input takes {}; a feed is \
                         conformed between rgba and yuv420p only",
                        to.pix_fmt
                    );
                }
                let incoming = crate::frame_len_for(pix_fmt, width, height)?;
                conn.demux.set_stream_limit(index, incoming as u64);
                let picture = Picture {
                    width,
                    height,
                    pix_fmt,
                    colour: (pix_fmt != "rgba")
                        .then(|| frame_colour(&stream.media))
                        .transpose()?,
                    to: *to,
                    to_colour: (to.pix_fmt != "rgba")
                        .then(|| port_colour(to.color.as_ref()))
                        .transpose()?,
                };
                if !picture.same() {
                    report(format!(
                        "feed: {}: the feeder's picture is {width}x{height} {pix_fmt}, conformed \
                         to {}x{} {}",
                        spec.names(),
                        to.width,
                        to.height,
                        to.pix_fmt
                    ));
                }
                Convert::Picture(picture)
            }
            StreamFormat::Audio(to) => {
                let Media::Audio {
                    sample_rate,
                    channels,
                } = stream.media
                else {
                    continue;
                };
                let refused = if sample_rate != to.sample_rate {
                    Some(format!(
                        "the feeder's sound is {sample_rate} Hz and this input is {} Hz",
                        to.sample_rate
                    ))
                } else if channels != to.channels {
                    Some(format!(
                        "the feeder's sound has {channels} channels and this input has {}",
                        to.channels
                    ))
                } else {
                    None
                };
                let sample_fmt = stream.sample_fmt();
                let refused = refused.or_else(|| {
                    sample_fmt.is_none().then(|| {
                        format!(
                            "the feeder's sound is {} and a feed is raw PCM; send it with -c:a \
                             pcm_s16le",
                            stream.fourcc_name()
                        )
                    })
                });
                if let Some(why) = refused {
                    report(format!(
                        "feed: {}: {why}; its sound is left out",
                        spec.names()
                    ));
                    continue;
                }
                conn.demux.set_stream_limit(index, MAX_AUDIO_PACKET);
                Convert::Sound(Sound {
                    sample_fmt: sample_fmt.expect("checked"),
                    to: *to,
                })
            }
            _ => continue,
        };
        taken[member] = true;
        conn.lanes[index] = Some(Lane {
            member,
            base: stream.time_base,
            convert,
        });
    }
    if !taken.iter().any(|t| *t) {
        anyhow::bail!("the feeder sends nothing this input takes");
    }
    Ok(())
}

/// What each member is told its source is: the wire's tags, the picture's
/// and the whole file's, and the stream's own time base.
fn sources(spec: &FeedSpec, conn: &Conn) -> Vec<(u32, SourceInfo)> {
    let picture = conn.lanes.iter().position(|l| {
        matches!(
            l,
            Some(Lane {
                convert: Convert::Picture(_),
                ..
            })
        )
    });
    let mut tags: Vec<(String, String)> = conn.demux.tags(None).to_vec();
    if let Some(index) = picture {
        for (key, value) in conn.demux.tags(Some(index)) {
            match tags.iter_mut().find(|(k, _)| k == key) {
                Some(slot) => slot.1 = value.clone(),
                None => tags.push((key.clone(), value.clone())),
            }
        }
    }
    let mut told = Vec::new();
    for (index, lane) in conn.lanes.iter().enumerate() {
        let Some(lane) = lane else { continue };
        let member = &spec.members[lane.member];
        let (kind, codec) = match &lane.convert {
            Convert::Picture(_) | Convert::Timing { audio: None } => ("video", "rawvideo"),
            Convert::Data => ("data", "json"),
            Convert::Timing { audio: Some(_) } => match &member.format {
                StreamFormat::Audio(a) if a.sample_fmt == "s16" => ("audio", "pcm_s16le"),
                _ => ("audio", "pcm_f32le"),
            },
            Convert::Sound(sound) => (
                "audio",
                if sound.to.sample_fmt == "s16" {
                    "pcm_s16le"
                } else {
                    "pcm_f32le"
                },
            ),
        };
        told.push((
            member.id,
            SourceInfo {
                connection: conn.serial,
                tags: tags.clone(),
                base: ffrwd_wasm_runtime::runtime::TimeBase {
                    num: lane.base.num,
                    den: lane.base.den,
                },
                info: StreamInfo {
                    index: index as u32,
                    kind: kind.to_string(),
                    codec: codec.to_string(),
                    duration: None,
                    tags: tags.clone(),
                },
            },
        ));
    }
    told.sort_by_key(|(_, source)| source.info.kind != "video");
    told
}

fn is_timeout(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut | io::ErrorKind::Interrupted
    )
}

/// How long a port still held by the last run's connections is tried for.
const BIND_WAIT: Duration = Duration::from_secs(2);

/// Binds the loopback port a feed is served on, trying for a while where
/// the last run's connections still hold it.
pub fn bind(port: u16) -> std::io::Result<TcpListener> {
    let address = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    let until = Instant::now() + BIND_WAIT;
    loop {
        match TcpListener::bind(address) {
            Ok(listener) => return Ok(listener),
            Err(_) if Instant::now() < until => thread::sleep(POLL),
            Err(e) => return Err(e),
        }
    }
}

/// Serves `spec` until the run stops, on `bound` where the port was bound
/// before the run's inputs were read.
pub fn serve(
    spec: FeedSpec,
    bound: Option<TcpListener>,
    scheduler: Arc<Scheduler>,
    carried: &Carried,
) -> Result<()> {
    let listener = match bound {
        Some(listener) => listener,
        None => bind(spec.port).with_context(|| {
            format!(
                "feed: {}: 127.0.0.1:{} cannot be bound",
                spec.names(),
                spec.port
            )
        })?,
    };
    listener.set_nonblocking(true)?;
    let mut report = |line: String| eprintln!("{line}");
    report(format!(
        "feed: {} listens on 127.0.0.1:{}",
        spec.names(),
        spec.port
    ));
    let row = json!({
        "kind": "listen",
        "node": spec.node,
        "input": spec.members[0].name,
        "port": spec.port,
    });
    report(format!("{ROW_PREFIX}{row}"));
    let mut serial = 0u64;
    let mut current: Option<Conn> = None;
    let mut chunk = vec![0u8; READ_CHUNK];
    loop {
        if scheduler.stopped() {
            return Ok(());
        }
        if let Some(conn) = current.as_mut() {
            let outcome = read_on(&spec, conn, &scheduler, &mut chunk, &mut report, carried);
            match outcome {
                Turn::Stopped => return Ok(()),
                Turn::Closed(why) => {
                    let conn = current.take().expect("current");
                    report(format!("feed: {}: {why}", spec.names()));
                    if conn.opened {
                        for member in &spec.members {
                            if !scheduler.source_close(member.id) {
                                return Ok(());
                            }
                        }
                    }
                }
                Turn::Open => {}
            }
        }
        match listener.accept() {
            Ok((socket, _)) => {
                let silent = current.as_ref().is_some_and(|c| c.bytes == 0);
                if current.is_none() || silent {
                    serial += 1;
                    let granted = socket2::SockRef::from(&socket)
                        .set_recv_buffer_size(RECEIVE_BUFFER)
                        .and_then(|_| socket2::SockRef::from(&socket).recv_buffer_size())
                        .unwrap_or(0);
                    report(format!(
                        "feed: {}: a feeder connected on 127.0.0.1:{} (receive buffer {granted} bytes)",
                        spec.names(),
                        spec.port
                    ));
                    current = Some(Conn::new(serial, socket, spec.members.len())?);
                } else {
                    drop(socket);
                    report(format!(
                        "feed: {}: a second feeder on 127.0.0.1:{} was refused; one at a time",
                        spec.names(),
                        spec.port
                    ));
                }
            }
            Err(e) if is_timeout(&e) => {}
            Err(e) => return Err(e).context("accepting a feeder"),
        }
        if current.is_none() {
            thread::sleep(POLL);
        }
    }
}

/// What one turn of reading a connection came to.
enum Turn {
    Open,
    Closed(String),
    Stopped,
}

/// One read off the connection, once its input has room: the bytes into
/// the demuxer, and everything they make on to the lanes.
fn read_on(
    spec: &FeedSpec,
    conn: &mut Conn,
    scheduler: &Scheduler,
    chunk: &mut [u8],
    report: &mut dyn FnMut(String),
    carried: &Carried,
) -> Turn {
    if conn.opened {
        match scheduler.wait_room(spec.members[0].id, POLL) {
            Room::Stopped => return Turn::Stopped,
            Room::Full => return Turn::Open,
            Room::Open => {}
        }
    }
    match conn.socket.read(chunk) {
        Ok(0) if conn.bytes == 0 => {
            Turn::Closed("a connection that sent nothing closed".to_string())
        }
        Ok(0) => Turn::Closed("the feeder finished".to_string()),
        Ok(n) => {
            conn.bytes += n as u64;
            conn.demux.feed(&chunk[..n]);
            match drain(spec, conn, scheduler, report, carried) {
                Ok(true) => Turn::Open,
                Ok(false) => Turn::Stopped,
                Err(e) => Turn::Closed(format!("the feeder's stream was refused: {e:#}")),
            }
        }
        Err(e) if is_timeout(&e) => Turn::Open,
        Err(e) => Turn::Closed(format!("the connection broke: {e}")),
    }
}

/// Everything the connection's bytes make: its headers matched once they
/// are whole, every frame on to its member. False once the run has stopped.
fn drain(
    spec: &FeedSpec,
    conn: &mut Conn,
    scheduler: &Scheduler,
    report: &mut dyn FnMut(String),
    carried: &Carried,
) -> Result<bool> {
    loop {
        let event = match conn.demux.next_event()? {
            Some(event) => event,
            None => return Ok(true),
        };
        match event {
            Event::EndOfHeaders if !conn.opened => {
                match_streams(spec, conn, report)?;
                conn.opened = true;
                for (id, source) in sources(spec, conn) {
                    if !scheduler.source_open(id, source) {
                        return Ok(false);
                    }
                }
            }
            Event::Frame { stream, packet } => {
                let Some(lane) = conn.lanes.get(stream).and_then(Option::as_ref) else {
                    continue;
                };
                let payload = conn.demux.payload();
                let member = &spec.members[lane.member];
                if let Convert::Data = lane.convert {
                    if crate::heartbeat::is_heartbeat(payload) {
                        continue;
                    }
                    let message = Message {
                        pts: packet.pts,
                        data: payload.to_vec(),
                    };
                    if !scheduler.arrive(member.id, Item::Message(message)) {
                        return Ok(false);
                    }
                    continue;
                }
                if !matches!(lane.convert, Convert::Timing { .. }) {
                    carried.conformed.fetch_add(1, Ordering::Relaxed);
                    carried
                        .bytes
                        .fetch_add(payload.len() as u64, Ordering::Relaxed);
                }
                let frame = match &lane.convert {
                    Convert::Data => continue,
                    Convert::Timing { audio } => TickFrame {
                        pts: packet.pts,
                        duration: audio.map(|width| (payload.len() / width.max(1)) as i64),
                        data: Arc::new(Vec::new()),
                        rows: Vec::new(),
                    },
                    Convert::Picture(picture) => TickFrame {
                        pts: packet.pts,
                        duration: None,
                        data: Arc::new(picture.convert(payload)?),
                        rows: Vec::new(),
                    },
                    Convert::Sound(sound) => {
                        let data = sound.convert(payload);
                        let samples = data.len() / sound.to.sample_len().max(1);
                        TickFrame {
                            pts: packet.pts,
                            duration: Some(samples as i64),
                            data: Arc::new(data),
                            rows: Vec::new(),
                        }
                    }
                };
                if !scheduler.arrive(member.id, Item::Frame(frame)) {
                    return Ok(false);
                }
            }
            Event::EndOfInput => return Ok(true),
            _ => {}
        }
    }
}
