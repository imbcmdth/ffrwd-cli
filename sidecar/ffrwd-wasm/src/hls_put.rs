//! An HLS destination written through the relay.
//!
//! ffmpeg's hls muxer writes a subtitle rendition beside the picture and the
//! sound, its WebVTT segments cut where the media segments are cut, but it
//! writes no `X-TIMESTAMP-MAP` into them, and without one a player maps a cue
//! time of zero to a media timestamp of zero. An MPEG-TS segment's timestamps
//! start 1.4 s later than the cues' (the muxer's own delay, twice
//! `-muxdelay`), so the captions would show that much early. The muxer
//! writes over HTTP as readily as to files: it PUTs each file, a segment as
//! it is cut and a playlist each time it changes, and DELETEs one that has
//! left the playlist. The relay takes those requests and writes each file
//! where its path says, each WebVTT segment with the header line that maps
//! its cue times onto the media's, and each subtitle rendition of a playlist
//! marked `AUTOSELECT=YES`, which the muxer has no way to write.
//!
//! A request's path is `/<key>/<path>`: the key names the destination, and
//! with it the MPEG-TS time a cue time of zero lands on; the path is the file
//! as ffmpeg would have written it, relative to the relay's own directory
//! unless it is absolute. A key the host did not give is refused.
//!
//! A file goes into place whole, written beside itself and renamed, so a
//! reader never sees part of one. A playlist goes into place once every file
//! it names that is still arriving has, so it never names a segment a player
//! cannot fetch yet.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

/// The longest request head read, its request line and headers together.
const HEAD_MOST: usize = 64 << 10;
/// The largest playlist or WebVTT segment held in memory to be rewritten.
const TEXT_MOST: usize = 16 << 20;
/// How long a playlist waits for a file it names that is still arriving.
const NAMED_WAIT: Duration = Duration::from_secs(10);
/// How long a rename that the platform refused is tried again for: on
/// Windows a file being read cannot be replaced until its reader lets go.
const RENAME_WAIT: Duration = Duration::from_secs(2);
const RENAME_AGAIN: Duration = Duration::from_millis(25);
/// How long the relay, told to end, waits for requests still being written:
/// less than the five seconds its host gives it to exit.
pub const DRAIN_WAIT: Duration = Duration::from_secs(4);
/// How long no request may be arriving before a drain counts as done.
const DRAIN_QUIET: Duration = Duration::from_millis(250);

/// What the relay serves for HLS destinations, and what it is writing now.
pub struct Endpoint {
    address: SocketAddr,
    /// Each destination's key and the MPEG-TS time, in 90 kHz ticks, that
    /// its WebVTT segments map a cue time of zero to.
    keys: Mutex<HashMap<String, u64>>,
    /// The files whose bodies are still arriving.
    arriving: Mutex<HashSet<PathBuf>>,
    arrived: Condvar,
    /// Requests being answered right now.
    busy: AtomicUsize,
    /// When the last request started, for a drain.
    last_start: Mutex<Instant>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Endpoint {
    /// Opens the endpoint on loopback, on a port the system picks, and
    /// starts answering. `failed` hears each request that could not be
    /// written, since ffmpeg does not wait for the answer to a PUT.
    pub fn open(failed: fn(&str)) -> io::Result<Arc<Endpoint>> {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let endpoint = Arc::new(Endpoint {
            address: listener.local_addr()?,
            keys: Mutex::new(HashMap::new()),
            arriving: Mutex::new(HashSet::new()),
            arrived: Condvar::new(),
            busy: AtomicUsize::new(0),
            last_start: Mutex::new(Instant::now()),
        });
        let serving = Arc::clone(&endpoint);
        thread::Builder::new()
            .name("relay-hls".to_string())
            .spawn(move || {
                for socket in listener.incoming() {
                    let Ok(socket) = socket else {
                        thread::sleep(Duration::from_millis(50));
                        continue;
                    };
                    serving.busy.fetch_add(1, Ordering::SeqCst);
                    *lock(&serving.last_start) = Instant::now();
                    let answering = Arc::clone(&serving);
                    let spawned = thread::Builder::new()
                        .name("relay-hls-put".to_string())
                        .spawn(move || {
                            if let Err(e) = answering.answer(socket) {
                                failed(&e);
                            }
                            answering.busy.fetch_sub(1, Ordering::SeqCst);
                        });
                    if spawned.is_err() {
                        serving.busy.fetch_sub(1, Ordering::SeqCst);
                    }
                }
            })?;
        Ok(endpoint)
    }

    pub fn address(&self) -> SocketAddr {
        self.address
    }

    /// Takes requests for `key`, whose WebVTT segments map a cue time of
    /// zero to `mpegts`.
    pub fn add(&self, key: String, mpegts: u64) {
        lock(&self.keys).insert(key, mpegts);
    }

    /// Waits for the requests still being answered, and for any a client
    /// sent just before, up to `wait`.
    pub fn drain(&self, wait: Duration) {
        let deadline = Instant::now() + wait;
        while Instant::now() < deadline {
            let quiet = lock(&self.last_start).elapsed() >= DRAIN_QUIET;
            if quiet && self.busy.load(Ordering::SeqCst) == 0 {
                return;
            }
            thread::sleep(Duration::from_millis(20));
        }
    }

    /// Answers one connection: one request, then the connection closes, as
    /// the hls muxer asks.
    fn answer(&self, socket: TcpStream) -> Result<(), String> {
        let mut writer = socket
            .try_clone()
            .map_err(|e| format!("answering an HLS request: {e}"))?;
        let mut reader = BufReader::new(socket);
        let result = self.request(&mut reader);
        let status = match &result {
            Ok(status) => *status,
            Err(failure) => failure.status,
        };
        let reply = format!("HTTP/1.1 {status}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        let _ = writer.write_all(reply.as_bytes());
        let _ = writer.flush();
        result.map(drop).map_err(|failure| failure.message)
    }

    fn request(&self, reader: &mut impl BufRead) -> Result<&'static str, Failure> {
        let head = Head::read(reader)?;
        let (key, rest) = head
            .target
            .strip_prefix('/')
            .and_then(|target| target.split_once('/'))
            .ok_or_else(|| Failure::new("404 Not Found", "an HLS request names no destination"))?;
        let Some(mpegts) = lock(&self.keys).get(key).copied() else {
            return Err(Failure::new(
                "404 Not Found",
                "an HLS request names a destination the relay was not given",
            ));
        };
        let path = local_path(rest);
        match head.method.as_str() {
            "PUT" | "POST" => {
                self.put(&path, &head, reader, mpegts)?;
                Ok("201 Created")
            }
            "DELETE" => {
                // A body it declares is read and dropped: a socket closed with
                // bytes unread is reset on Linux, and the answer with it.
                if head.chunked || head.length.is_some() {
                    head.copy_body(reader, &mut io::sink())?;
                }
                match fs::remove_file(&path) {
                    Ok(()) => {}
                    Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                    Err(e) => {
                        return Err(Failure::new(
                            "500 Internal Server Error",
                            &format!("removing {}: {e}", path.display()),
                        ))
                    }
                }
                Ok("204 No Content")
            }
            other => Err(Failure::new(
                "405 Method Not Allowed",
                &format!("an HLS request is a PUT or a DELETE, and this one is a {other}"),
            )),
        }
    }

    fn put(
        &self,
        path: &Path,
        head: &Head,
        reader: &mut impl BufRead,
        mpegts: u64,
    ) -> Result<(), Failure> {
        let writing = |e: io::Error| {
            Failure::new(
                "500 Internal Server Error",
                &format!("writing {}: {e}", path.display()),
            )
        };
        lock(&self.arriving).insert(path.to_path_buf());
        let result = (|| {
            if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                fs::create_dir_all(parent).map_err(writing)?;
            }
            let partial = partial_path(path);
            let kind = Kind::of(path);
            {
                let mut file = File::create(&partial).map_err(writing)?;
                match kind {
                    Kind::Media => head.copy_body(reader, &mut file)?,
                    Kind::Playlist | Kind::Cues => {
                        let mut body = Vec::new();
                        head.copy_body(reader, &mut body)?;
                        let body = if kind == Kind::Cues {
                            timestamp_mapped(&body, mpegts)
                        } else {
                            self.wait_for_named(path, &body);
                            subtitles_autoselected(&body)
                        };
                        file.write_all(&body).map_err(writing)?;
                    }
                }
                file.flush().map_err(writing)?;
            }
            rename(&partial, path).map_err(writing)
        })();
        let mut arriving = lock(&self.arriving);
        arriving.remove(path);
        self.arrived.notify_all();
        result
    }

    /// Waits until no file the playlist `body` names is still arriving.
    fn wait_for_named(&self, playlist: &Path, body: &[u8]) {
        let base = playlist.parent().unwrap_or(Path::new(""));
        let named: Vec<PathBuf> = named_uris(&String::from_utf8_lossy(body))
            .into_iter()
            .map(|uri| normal(&base.join(uri)))
            .collect();
        let deadline = Instant::now() + NAMED_WAIT;
        let mut arriving = lock(&self.arriving);
        while named.iter().any(|path| arriving.contains(path)) {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return;
            }
            arriving = self
                .arrived
                .wait_timeout(arriving, left)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }
}

/// A request that could not be answered: the status, and why.
struct Failure {
    status: &'static str,
    message: String,
}

impl Failure {
    fn new(status: &'static str, message: &str) -> Failure {
        Failure {
            status,
            message: message.to_string(),
        }
    }

    fn bad(message: &str) -> Failure {
        Failure::new("400 Bad Request", message)
    }
}

/// A request's line and the headers that say how its body arrives.
struct Head {
    method: String,
    target: String,
    chunked: bool,
    length: Option<u64>,
}

impl Head {
    fn read(reader: &mut impl BufRead) -> Result<Head, Failure> {
        let mut lines = Vec::new();
        let mut read = 0usize;
        loop {
            let mut line = Vec::new();
            let n = reader
                .read_until(b'\n', &mut line)
                .map_err(|e| Failure::bad(&format!("reading an HLS request: {e}")))?;
            read += n;
            if n == 0 || read > HEAD_MOST {
                return Err(Failure::bad("an HLS request ended before its headers did"));
            }
            let line = String::from_utf8_lossy(&line).trim_end().to_string();
            if line.is_empty() {
                break;
            }
            lines.push(line);
        }
        let request = lines
            .first()
            .ok_or_else(|| Failure::bad("an HLS request with no request line"))?;
        // The target is everything between the method and the version, as
        // ffmpeg sends a path: unescaped, spaces and all.
        let (method, rest) = request
            .split_once(' ')
            .ok_or_else(|| Failure::bad("an HLS request line with no target"))?;
        let (target, _version) = rest
            .rsplit_once(' ')
            .ok_or_else(|| Failure::bad("an HLS request line with no version"))?;
        let mut head = Head {
            method: method.to_ascii_uppercase(),
            target: target.to_string(),
            chunked: false,
            length: None,
        };
        for line in &lines[1..] {
            let Some((name, value)) = line.split_once(':') else {
                continue;
            };
            let value = value.trim();
            match name.trim().to_ascii_lowercase().as_str() {
                "transfer-encoding" => {
                    head.chunked = value.to_ascii_lowercase().contains("chunked")
                }
                "content-length" => {
                    head.length = Some(
                        value
                            .parse()
                            .map_err(|_| Failure::bad("an HLS request's length is not a number"))?,
                    )
                }
                _ => {}
            }
        }
        Ok(head)
    }

    /// Copies the body into `out`: chunk by chunk, the stated length, or
    /// everything up to the connection's end.
    fn copy_body(&self, reader: &mut impl BufRead, out: &mut impl Write) -> Result<(), Failure> {
        let cut_short = |e: io::Error| Failure::bad(&format!("reading an HLS request's body: {e}"));
        let copy = |reader: &mut dyn Read, out: &mut dyn Write| -> Result<u64, Failure> {
            io::copy(reader, out).map_err(cut_short)
        };
        if self.chunked {
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).map_err(cut_short)?;
                let size = line.trim().split(';').next().unwrap_or("");
                let size = u64::from_str_radix(size, 16)
                    .map_err(|_| Failure::bad("an HLS request's chunk has no size"))?;
                if size == 0 {
                    // The trailer, up to its blank line.
                    loop {
                        let mut trailer = String::new();
                        let n = reader.read_line(&mut trailer).map_err(cut_short)?;
                        if n == 0 || trailer.trim().is_empty() {
                            return Ok(());
                        }
                    }
                }
                let copied = copy(&mut reader.by_ref().take(size), out)?;
                if copied != size {
                    return Err(Failure::bad("an HLS request ended inside a chunk"));
                }
                let mut end = String::new();
                reader.read_line(&mut end).map_err(cut_short)?;
            }
        }
        match self.length {
            Some(length) => {
                let copied = copy(&mut reader.by_ref().take(length), out)?;
                if copied != length {
                    return Err(Failure::bad("an HLS request ended before its body did"));
                }
            }
            None => {
                copy(reader, out)?;
            }
        }
        Ok(())
    }
}

/// What a file is, by its name, which is how it is written.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Playlist,
    Cues,
    Media,
}

impl Kind {
    fn of(path: &Path) -> Kind {
        match path.extension().and_then(|e| e.to_str()) {
            Some(ext) if ext.eq_ignore_ascii_case("m3u8") => Kind::Playlist,
            Some(ext) if ext.eq_ignore_ascii_case("vtt") => Kind::Cues,
            _ => Kind::Media,
        }
    }
}

/// A request's path as a file: relative to the relay's directory unless it
/// is absolute, as `/data/x` or `C:/x` is.
fn local_path(rest: &str) -> PathBuf {
    normal(Path::new(rest))
}

/// `path` with its `.` steps gone and each `..` taking the step before it,
/// so two spellings of one file compare equal.
fn normal(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Where a file is written before it is renamed into place: beside it,
/// hidden.
fn partial_path(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    path.with_file_name(format!(".{name}.part"))
}

fn rename(from: &Path, to: &Path) -> io::Result<()> {
    let deadline = Instant::now() + RENAME_WAIT;
    loop {
        match fs::rename(from, to) {
            Ok(()) => return Ok(()),
            Err(e) if Instant::now() >= deadline => {
                let _ = fs::remove_file(from);
                return Err(e);
            }
            Err(_) => thread::sleep(RENAME_AGAIN),
        }
    }
}

/// Every file a playlist names by a relative URI: its segment and playlist
/// lines, and the `URI` of its tags.
fn named_uris(playlist: &str) -> Vec<String> {
    let mut uris = Vec::new();
    for line in playlist.lines().map(str::trim) {
        if line.is_empty() {
            continue;
        }
        if line.starts_with('#') {
            if let Some((_, after)) = line.split_once("URI=\"") {
                if let Some((uri, _)) = after.split_once('"') {
                    uris.push(uri.to_string());
                }
            }
            continue;
        }
        uris.push(line.to_string());
    }
    uris.retain(|uri| !uri.contains("://"));
    uris
}

/// A WebVTT segment with `X-TIMESTAMP-MAP` in its header: a cue time of zero
/// is the MPEG-TS time `mpegts`. One that already has a map is left alone.
fn timestamp_mapped(body: &[u8], mpegts: u64) -> Vec<u8> {
    let text = String::from_utf8_lossy(body);
    let Some(first) = text.lines().next() else {
        return body.to_vec();
    };
    if !first.trim_start_matches('\u{feff}').starts_with("WEBVTT") {
        return body.to_vec();
    }
    let header_has_map = text
        .lines()
        .skip(1)
        .take_while(|line| !line.trim().is_empty())
        .any(|line| line.starts_with("X-TIMESTAMP-MAP="));
    if header_has_map || body.len() > TEXT_MOST {
        return body.to_vec();
    }
    let rest = &text[first.len()..];
    let rest = rest
        .strip_prefix("\r\n")
        .or_else(|| rest.strip_prefix('\n'))
        .unwrap_or(rest);
    format!("{first}\nX-TIMESTAMP-MAP=MPEGTS:{mpegts},LOCAL:00:00:00.000\n{rest}").into_bytes()
}

/// A playlist whose subtitle renditions each say `AUTOSELECT=YES`, so a
/// player picks the one in its viewer's language; ffmpeg's muxer writes no
/// `AUTOSELECT`, and a rendition that is the `DEFAULT` has to be one a player
/// may select by itself.
fn subtitles_autoselected(body: &[u8]) -> Vec<u8> {
    let text = String::from_utf8_lossy(body);
    if !text.contains("TYPE=SUBTITLES") {
        return body.to_vec();
    }
    let mut out = String::with_capacity(text.len() + 64);
    for line in text.split_inclusive('\n') {
        let media = line.starts_with("#EXT-X-MEDIA:")
            && line.contains("TYPE=SUBTITLES")
            && !line.contains("AUTOSELECT=");
        match line.find(",URI=").filter(|_| media) {
            Some(at) => {
                out.push_str(&line[..at]);
                out.push_str(",AUTOSELECT=YES");
                out.push_str(&line[at..]);
            }
            None => out.push_str(line),
        }
    }
    out.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn head(text: &str) -> Head {
        Head::read(&mut Cursor::new(text.as_bytes())).unwrap_or_else(|e| panic!("{}", e.message))
    }

    #[test]
    fn a_segment_gets_its_map_after_the_signature() {
        let body = b"WEBVTT\n\n00:01.067 --> 00:02.567\n<v Ann>hello\n";
        assert_eq!(
            String::from_utf8(timestamp_mapped(body, 126000)).expect("text"),
            "WEBVTT\nX-TIMESTAMP-MAP=MPEGTS:126000,LOCAL:00:00:00.000\n\n\
             00:01.067 --> 00:02.567\n<v Ann>hello\n"
        );
        // A segment with no cue still says where its cue times lie.
        assert_eq!(
            String::from_utf8(timestamp_mapped(b"WEBVTT\n", 0)).expect("text"),
            "WEBVTT\nX-TIMESTAMP-MAP=MPEGTS:0,LOCAL:00:00:00.000\n"
        );
        let mapped = b"WEBVTT\nX-TIMESTAMP-MAP=MPEGTS:900000,LOCAL:00:00:00.000\n\n";
        assert_eq!(timestamp_mapped(mapped, 126000), mapped.to_vec());
        assert_eq!(
            timestamp_mapped(b"not webvtt", 126000),
            b"not webvtt".to_vec()
        );
    }

    #[test]
    fn a_subtitle_rendition_is_one_a_player_may_pick() {
        let master = "#EXTM3U\n\
            #EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID=\"subs\",NAME=\"eng\",DEFAULT=YES,LANGUAGE=\"eng\",URI=\"240p/index_vtt.m3u8\"\n\
            #EXT-X-STREAM-INF:BANDWIDTH=1,SUBTITLES=\"subs\"\n240p/index.m3u8\n";
        assert_eq!(
            String::from_utf8(subtitles_autoselected(master.as_bytes())).expect("text"),
            master.replace(
                ",URI=\"240p/index_vtt",
                ",AUTOSELECT=YES,URI=\"240p/index_vtt"
            )
        );
        let once = subtitles_autoselected(master.as_bytes());
        assert_eq!(subtitles_autoselected(&once), once);
    }

    #[test]
    fn a_playlist_names_its_segments_and_the_uris_of_its_tags() {
        let playlist = "#EXTM3U\n#EXT-X-MAP:URI=\"init.mp4\"\n#EXTINF:2.0,\nsegment_0.m4s\n\
            #EXT-X-MEDIA:TYPE=SUBTITLES,URI=\"240p/index_vtt.m3u8\"\nhttp://elsewhere/x.ts\n";
        assert_eq!(
            named_uris(playlist),
            vec!["init.mp4", "segment_0.m4s", "240p/index_vtt.m3u8"]
        );
    }

    #[test]
    fn a_request_line_keeps_a_path_with_spaces() {
        let read =
            head("PUT /k/out dir/240p/index0.vtt HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n");
        assert_eq!(read.method, "PUT");
        assert_eq!(read.target, "/k/out dir/240p/index0.vtt");
        assert!(read.chunked);
    }

    #[test]
    fn a_chunked_body_is_copied_whole() {
        let read = head("PUT /k/a.ts HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n");
        let mut body = Cursor::new(b"5\r\nhello\r\n3;x=y\r\n, w\r\n0\r\n\r\n".to_vec());
        let mut out = Vec::new();
        read.copy_body(&mut body, &mut out)
            .unwrap_or_else(|e| panic!("{}", e.message));
        assert_eq!(out, b"hello, w");
    }

    #[test]
    fn a_path_is_the_file_it_names() {
        assert_eq!(
            local_path("out/./240p/../master.m3u8"),
            PathBuf::from("out/master.m3u8")
        );
        assert_eq!(
            normal(&Path::new("out/240p").join("index0.vtt")),
            PathBuf::from("out/240p/index0.vtt")
        );
    }

    #[test]
    fn files_arrive_whole_and_a_playlist_after_what_it_names() {
        let dir = std::env::temp_dir().join(format!("ffrwd-hls-put-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let endpoint = Endpoint::open(|_| {}).expect("an endpoint");
        endpoint.add("k".to_string(), 126000);
        let root = dir.to_string_lossy().replace('\\', "/");
        let send = |method: &str, name: &str, body: &str| {
            let mut socket = TcpStream::connect(endpoint.address()).expect("connected");
            let chunk = if body.is_empty() {
                String::new()
            } else {
                format!("{:x}\r\n{body}\r\n", body.len())
            };
            write!(
                socket,
                "{method} /k/{root}/{name} HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n{chunk}0\r\n\r\n"
            )
            .expect("sent");
            let mut reply = String::new();
            socket.read_to_string(&mut reply).expect("answered");
            reply
        };
        assert!(send("PUT", "240p/index0.vtt", "WEBVTT\n").starts_with("HTTP/1.1 201"));
        assert!(
            send("PUT", "240p/index_vtt.m3u8", "#EXTM3U\nindex0.vtt\n").starts_with("HTTP/1.1 201")
        );
        assert_eq!(
            fs::read_to_string(dir.join("240p/index0.vtt")).expect("written"),
            "WEBVTT\nX-TIMESTAMP-MAP=MPEGTS:126000,LOCAL:00:00:00.000\n"
        );
        assert!(dir.join("240p/index_vtt.m3u8").exists());
        assert!(!dir.join("240p/.index0.vtt.part").exists());
        assert!(send("DELETE", "240p/index0.vtt", "").starts_with("HTTP/1.1 204"));
        assert!(!dir.join("240p/index0.vtt").exists());
        let mut socket = TcpStream::connect(endpoint.address()).expect("connected");
        write!(
            socket,
            "PUT /other/{root}/x.ts HTTP/1.1\r\nContent-Length: 0\r\n\r\n"
        )
        .expect("sent");
        let mut reply = String::new();
        socket.read_to_string(&mut reply).expect("answered");
        assert!(reply.starts_with("HTTP/1.1 404"), "got {reply}");
        let _ = fs::remove_dir_all(&dir);
    }
}
