//! Outbound network policy, read from `FFRWD_NET_POLICY`. Unset leaves the
//! two grants (`-udp`, `-http`) unrestricted; `public` refuses destinations
//! in the private, loopback, link-local, carrier-NAT, multicast and
//! broadcast ranges, except the ones `FFRWD_REACH_ALLOW` names. Local binds
//! are never restricted.
//!
//! For wasi:sockets the check runs per address use. For wasi:http the
//! request path resolves the authority itself and connects only to public
//! or allowed addresses, so a hostname is judged by what it resolves to,
//! not by its spelling, unless the allow list names it.

use std::future::Future;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, ToSocketAddrs};
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::task::{ready, Context, Poll};
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use http::uri::Scheme;
use http_body_util::BodyExt;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::TcpStream;
use wasmtime_wasi::sockets::SocketAddrUse;
use wasmtime_wasi_http::io::TokioIo;
use wasmtime_wasi_http::{default_send_request, Error, RequestOptions, WasiBody, WasiHttpHooks};

/// What `FFRWD_NET_POLICY` allows a granted module to reach.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NetPolicy {
    /// The variable is unset or empty: no restriction.
    Unrestricted,
    /// Public destinations, and the ones `FFRWD_REACH_ALLOW` names.
    Public(&'static ReachAllow),
}

/// One entry of `FFRWD_REACH_ALLOW`, as written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReachEntry {
    /// A host name, standing for the addresses it resolves to.
    Name(String),
    /// One address.
    Addr(IpAddr),
    /// Every address whose first `bits` bits are `net`'s.
    Cidr { net: IpAddr, bits: u8 },
}

/// The destinations `FFRWD_REACH_ALLOW` lets a module reach under `public`
/// although they are not public.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReachAllow {
    entries: Vec<ReachEntry>,
    /// What the host-name entries resolved to when the policy was read.
    resolved: Vec<IpAddr>,
}

/// `public` with no exceptions.
pub static NO_EXCEPTIONS: ReachAllow = ReachAllow {
    entries: Vec::new(),
    resolved: Vec::new(),
};

impl ReachAllow {
    /// Resolves every host-name entry once through `resolve`, which gives a
    /// name's addresses or none. A name with none warns on stderr and allows
    /// nothing.
    pub fn resolved(mut self, resolve: impl Fn(&str) -> Vec<IpAddr>) -> ReachAllow {
        for entry in &self.entries {
            let ReachEntry::Name(name) = entry else {
                continue;
            };
            let addrs = resolve(name);
            if addrs.is_empty() {
                eprintln!(
                    "ffrwd-wasm: FFRWD_REACH_ALLOW names {name:?}, which does not resolve here, \
                     so it allows nothing"
                );
            }
            self.resolved
                .extend(addrs.into_iter().map(|addr| addr.to_canonical()));
        }
        self
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether `addr` is inside an entry: an address or prefix holding it, or
    /// a host name that resolved to it.
    fn permits(&self, addr: IpAddr) -> bool {
        let addr = addr.to_canonical();
        self.resolved.contains(&addr)
            || self.entries.iter().any(|entry| match entry {
                ReachEntry::Name(_) => false,
                ReachEntry::Addr(allowed) => *allowed == addr,
                ReachEntry::Cidr { net, bits } => in_prefix(addr, *net, *bits),
            })
    }

    /// Whether `host` is a host-name entry, in any case, with or without a
    /// trailing dot.
    fn names(&self, host: &str) -> bool {
        self.entries
            .iter()
            .any(|entry| matches!(entry, ReachEntry::Name(name) if same_name(name, host)))
    }

    /// What `public` allows, for a refusal line: the list is mentioned only
    /// when it is set.
    fn allowed_scope(&self) -> &'static str {
        if self.is_empty() {
            "public destinations only"
        } else {
            "public destinations and FFRWD_REACH_ALLOW only"
        }
    }
}

/// Whether `addr`'s first `bits` bits are `net`'s. Families never match
/// across.
fn in_prefix(addr: IpAddr, net: IpAddr, bits: u8) -> bool {
    match (addr, net) {
        (IpAddr::V4(addr), IpAddr::V4(net)) => {
            let mask = u32::MAX.checked_shl(32 - u32::from(bits)).unwrap_or(0);
            (u32::from(addr) & mask) == (u32::from(net) & mask)
        }
        (IpAddr::V6(addr), IpAddr::V6(net)) => {
            let mask = u128::MAX.checked_shl(128 - u32::from(bits)).unwrap_or(0);
            (u128::from(addr) & mask) == (u128::from(net) & mask)
        }
        _ => false,
    }
}

/// Whether two host names are the same name: case does not count, nor a
/// trailing dot.
fn same_name(a: &str, b: &str) -> bool {
    a.trim_end_matches('.')
        .eq_ignore_ascii_case(b.trim_end_matches('.'))
}

/// Whether `entry` is spelled like a host name: labels of letters, digits
/// and hyphens between dots, none empty, none beginning or ending with a
/// hyphen, and the last one holding a letter, which tells a mistyped
/// address from a name.
fn is_host_name(entry: &str) -> bool {
    let name = entry.strip_suffix('.').unwrap_or(entry);
    let label = |label: &str| {
        !label.is_empty()
            && label.len() <= 63
            && !label.starts_with('-')
            && !label.ends_with('-')
            && label
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-')
    };
    let last = name.rsplit('.').next().unwrap_or("");
    !name.is_empty()
        && name.len() <= 253
        && name.split('.').all(label)
        && last.bytes().any(|b| b.is_ascii_alphabetic())
}

/// Parses `FFRWD_NET_POLICY` and, under `public`, `FFRWD_REACH_ALLOW`:
/// `None` when the network is unrestricted, otherwise the exceptions
/// `public` makes, their host names not yet resolved. The policy is an exact
/// match and every entry a host name, an IP literal or a CIDR, trimmed,
/// commas between: a typo in either is an error, never an open network.
/// Unrestricted does not read the list.
pub fn parse(policy: Option<&str>, reach_allow: Option<&str>) -> Result<Option<ReachAllow>> {
    match policy {
        None | Some("") => Ok(None),
        Some("public") => parse_reach_allow(reach_allow).map(Some),
        Some(other) => bail!(
            "FFRWD_NET_POLICY is {other:?}: the only accepted value is \"public\"; \
             unset it to leave the network unrestricted"
        ),
    }
}

/// One `FFRWD_REACH_ALLOW` value. Unset or empty means no exceptions, and an
/// empty entry between commas is nothing.
fn parse_reach_allow(value: Option<&str>) -> Result<ReachAllow> {
    let entries = value
        .unwrap_or("")
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(parse_entry)
        .collect::<Result<Vec<_>>>()?;
    Ok(ReachAllow {
        entries,
        resolved: Vec::new(),
    })
}

/// One trimmed entry of `FFRWD_REACH_ALLOW`.
fn parse_entry(entry: &str) -> Result<ReachEntry> {
    if let Ok(addr) = entry.parse::<IpAddr>() {
        return Ok(ReachEntry::Addr(addr.to_canonical()));
    }
    if let Some((net, bits)) = entry.split_once('/') {
        if let (Ok(net), Ok(bits)) = (net.parse::<IpAddr>(), bits.parse::<u8>()) {
            let width = if net.is_ipv4() { 32 } else { 128 };
            if bits > width {
                bail!("FFRWD_REACH_ALLOW entry {entry:?} has a prefix longer than {width} bits");
            }
            return Ok(ReachEntry::Cidr { net, bits });
        }
    }
    if is_host_name(entry) {
        return Ok(ReachEntry::Name(entry.to_string()));
    }
    bail!(
        "FFRWD_REACH_ALLOW entry {entry:?} is not a host name, an IP address or a CIDR \
         like 10.0.0.0/8; unset the variable to allow public destinations only"
    )
}

/// The process's policy, read from the environment once; the allow list's
/// host names are resolved then.
pub fn net_policy() -> Result<NetPolicy> {
    static POLICY: OnceLock<std::result::Result<Option<ReachAllow>, String>> = OnceLock::new();
    match POLICY.get_or_init(|| read_policy().map_err(|e| e.to_string())) {
        Ok(None) => Ok(NetPolicy::Unrestricted),
        Ok(Some(allow)) => Ok(NetPolicy::Public(allow)),
        Err(message) => Err(anyhow!(message.clone())),
    }
}

fn read_policy() -> Result<Option<ReachAllow>> {
    let policy = env_var("FFRWD_NET_POLICY")?;
    let reach_allow = env_var("FFRWD_REACH_ALLOW")?;
    Ok(parse(policy.as_deref(), reach_allow.as_deref())?.map(|allow| allow.resolved(lookup)))
}

/// A variable's text, `None` when it is unset.
fn env_var(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            bail!("{name} is not unicode: set it to text or unset it")
        }
    }
}

/// Every address `name` resolves to through the system resolver, or none.
fn lookup(name: &str) -> Vec<IpAddr> {
    (name, 0)
        .to_socket_addrs()
        .map(|addrs| addrs.map(|addr| addr.ip()).collect())
        .unwrap_or_default()
}

/// Whether `addr` is a public destination: outside every private, loopback,
/// link-local, carrier-NAT, multicast and broadcast range. A v4-mapped v6
/// address is judged by the v4 address it carries.
pub fn public_destination(addr: IpAddr) -> bool {
    match addr {
        IpAddr::V4(v4) => public_v4(v4),
        IpAddr::V6(v6) => match v6.to_ipv4_mapped() {
            Some(v4) => public_v4(v4),
            None => {
                let head = v6.segments()[0];
                !(v6.is_unspecified()
                    || v6.is_loopback()
                    || (head & 0xffc0) == 0xfe80 // fe80::/10
                    || (head & 0xfe00) == 0xfc00 // fc00::/7
                    || (head & 0xff00) == 0xff00) // ff00::/8
            }
        },
    }
}

fn public_v4(addr: Ipv4Addr) -> bool {
    let [a, b, ..] = addr.octets();
    !(a == 0 // 0.0.0.0/8
        || a == 10 // 10.0.0.0/8
        || (a == 100 && (b & 0xc0) == 64) // 100.64.0.0/10
        || a == 127 // 127.0.0.0/8
        || (a == 169 && b == 254) // 169.254.0.0/16
        || (a == 172 && (b & 0xf0) == 16) // 172.16.0.0/12
        || (a == 192 && b == 168) // 192.168.0.0/16
        || (a & 0xf0) == 0xe0 // 224.0.0.0/4
        || addr == Ipv4Addr::BROADCAST)
}

/// Whether the sockets layer may use `addr` for `reason` under `policy`.
/// Binds and inbound addresses pass; only where traffic goes is policed, and
/// there a public destination or one the allow list holds passes.
pub fn permit_socket_addr(policy: NetPolicy, addr: SocketAddr, reason: SocketAddrUse) -> bool {
    match policy {
        NetPolicy::Unrestricted => true,
        NetPolicy::Public(allow) => match reason {
            SocketAddrUse::TcpBind
            | SocketAddrUse::TcpListen
            | SocketAddrUse::TcpAccept
            | SocketAddrUse::UdpBind
            | SocketAddrUse::UdpReceive => true,
            SocketAddrUse::TcpConnect | SocketAddrUse::UdpSend => {
                public_destination(addr.ip()) || allow.permits(addr.ip())
            }
        },
    }
}

/// The address check installed under `public`: permits or refuses, and says
/// so on stderr when refusing.
pub(crate) fn check_socket_addr(
    allow: &'static ReachAllow,
    addr: SocketAddr,
    reason: SocketAddrUse,
) -> bool {
    let allowed = permit_socket_addr(NetPolicy::Public(allow), addr, reason);
    if !allowed {
        let scope = allow.allowed_scope();
        eprintln!("ffrwd-wasm: refused {addr}: FFRWD_NET_POLICY=public allows {scope}");
    }
    allowed
}

type IoFuture = Box<dyn Future<Output = std::result::Result<(), Error>> + Send>;
type SendFuture = Box<
    dyn Future<Output = std::result::Result<(http::Response<WasiBody>, IoFuture), Error>> + Send,
>;

/// The store's wasi:http hooks. Every hook keeps its default; only how a
/// request is sent depends on the policy.
pub struct Hooks {
    policy: NetPolicy,
}

impl Hooks {
    pub fn new(policy: NetPolicy) -> Self {
        Self { policy }
    }
}

impl WasiHttpHooks for Hooks {
    fn send_request(
        &mut self,
        request: http::Request<WasiBody>,
        options: Option<RequestOptions>,
        fut: Box<dyn Future<Output = std::result::Result<(), Error>> + Send>,
    ) -> SendFuture {
        _ = fut;
        match self.policy {
            NetPolicy::Unrestricted => Box::new(async move {
                let (res, io) = default_send_request(request, options).await?;
                Ok((res.map(BodyExt::boxed_unsync), Box::new(io) as IoFuture))
            }),
            NetPolicy::Public(allow) => Box::new(send_request_public(allow, request, options)),
        }
    }
}

trait Stream: AsyncRead + AsyncWrite + Send + Sync + Unpin {}
impl<T> Stream for T where T: AsyncRead + AsyncWrite + Send + Sync + Unpin {}

/// Response body relaying `hyper`'s, timing out between frames the way the
/// crate's default request path does.
struct TimedBody {
    incoming: hyper::body::Incoming,
    timeout: tokio::time::Interval,
}

impl http_body::Body for TimedBody {
    type Data = <hyper::body::Incoming as http_body::Body>::Data;
    type Error = Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<std::result::Result<http_body::Frame<Self::Data>, Self::Error>>> {
        match Pin::new(&mut self.as_mut().incoming).poll_frame(cx) {
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Ready(Some(Err(err))) => {
                let err = if err.is_timeout() {
                    Error::HttpResponseTimeout
                } else {
                    Error::from(err)
                };
                Poll::Ready(Some(Err(err)))
            }
            Poll::Ready(Some(Ok(frame))) => {
                self.timeout.reset();
                Poll::Ready(Some(Ok(frame)))
            }
            Poll::Pending => {
                ready!(self.timeout.poll_tick(cx));
                Poll::Ready(Some(Err(Error::ConnectionReadTimeout)))
            }
        }
    }

    fn is_end_stream(&self) -> bool {
        self.incoming.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.incoming.size_hint()
    }
}

/// Sends `req` the way the crate's default request path does, except that
/// the authority is resolved here and only public or allowed addresses are
/// connected to. Resolving and connecting in one place is what closes the
/// gap between checking a name and dialing it.
async fn send_request_public(
    allow: &'static ReachAllow,
    mut req: http::Request<WasiBody>,
    options: Option<RequestOptions>,
) -> std::result::Result<(http::Response<WasiBody>, IoFuture), Error> {
    let uri = req.uri();
    let authority = uri.authority().ok_or(Error::HttpRequestUriInvalid)?;
    let use_tls = uri.scheme() == Some(&Scheme::HTTPS);
    let host = authority.host();
    let host = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host)
        .to_string();
    let port = authority
        .port_u16()
        .unwrap_or(if use_tls { 443 } else { 80 });

    let literal: Option<IpAddr> = host.parse().ok();
    let candidates: Vec<SocketAddr> = match literal {
        Some(ip) => vec![SocketAddr::new(ip, port)],
        None => tokio::net::lookup_host((host.as_str(), port))
            .await
            .map_err(|e| Error::DnsError {
                rcode: Some(e.to_string()),
                info_code: None,
            })?
            .collect(),
    };
    if candidates.is_empty() {
        return Err(Error::DnsError {
            rcode: Some("no address".to_string()),
            info_code: None,
        });
    }
    let dialable = dialable(allow, &host, candidates);
    if dialable.is_empty() {
        let scope = allow.allowed_scope();
        eprintln!(
            "ffrwd-wasm: refused http request to {host}:{port}: \
             FFRWD_NET_POLICY=public allows {scope}"
        );
        return Err(Error::DestinationIpProhibited);
    }

    let connect_timeout = options
        .and_then(|o| o.connect_timeout)
        .unwrap_or(Duration::from_secs(600));
    let first_byte_timeout = options
        .and_then(|o| o.first_byte_timeout)
        .unwrap_or(Duration::from_secs(600));
    let between_bytes_timeout = options
        .and_then(|o| o.between_bytes_timeout)
        .unwrap_or(Duration::from_secs(600));

    let mut stream = None;
    let mut failure = Error::ConnectionRefused;
    for addr in dialable {
        match tokio::time::timeout(connect_timeout, TcpStream::connect(addr)).await {
            Ok(Ok(connected)) => {
                stream = Some(connected);
                break;
            }
            Ok(Err(e)) => failure = Error::Connect(e),
            Err(_) => failure = Error::ConnectionTimeout,
        }
    }
    let Some(stream) = stream else {
        return Err(failure);
    };

    let stream: Box<dyn Stream> = if use_tls {
        let roots = rustls::RootCertStore {
            roots: webpki_roots::TLS_SERVER_ROOTS.into(),
        };
        let config = rustls::ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth();
        let connector = tokio_rustls::TlsConnector::from(Arc::new(config));
        // The certificate is verified against the requested name, not the
        // address dialed.
        let name = match literal {
            Some(ip) => rustls::pki_types::ServerName::from(ip),
            None => rustls::pki_types::ServerName::try_from(host.clone())
                .map_err(Error::InvalidDnsNameError)?,
        };
        Box::new(connector.connect(name, stream).await.map_err(Error::Tls)?)
    } else {
        Box::new(stream)
    };

    let (mut sender, conn) = tokio::time::timeout(
        connect_timeout,
        hyper::client::conn::http1::Builder::new().handshake(TokioIo::new(stream)),
    )
    .await
    .map_err(|_| Error::ConnectionTimeout)?
    .map_err(Error::from)?;

    // The request carries scheme and authority, which belong on the wire
    // only when addressing a proxy.
    *req.uri_mut() = http::Uri::builder()
        .path_and_query(
            req.uri()
                .path_and_query()
                .map(|p| p.as_str())
                .unwrap_or("/"),
        )
        .build()
        .expect("comes from a valid request");

    let conn = wasmtime_wasi::runtime::spawn(async move {
        conn.await.map_err(|err| {
            if err.is_timeout() {
                Error::HttpResponseTimeout
            } else {
                Error::from(err)
            }
        })
    });

    let res = tokio::time::timeout(first_byte_timeout, sender.send_request(req))
        .await
        .map_err(|_| Error::ConnectionReadTimeout)?
        .map_err(Error::from)?;
    let res = res.map(|incoming| {
        let mut timeout = tokio::time::interval(between_bytes_timeout);
        timeout.reset();
        TimedBody { incoming, timeout }.boxed_unsync()
    });
    Ok((res, Box::new(conn) as IoFuture))
}

/// Which of a request's addresses `public` may dial: every one when `host`
/// is a host-name entry of the allow list, otherwise the public and the
/// allowed ones.
fn dialable(allow: &ReachAllow, host: &str, candidates: Vec<SocketAddr>) -> Vec<SocketAddr> {
    if allow.names(host) {
        return candidates;
    }
    candidates
        .into_iter()
        .filter(|addr| public_destination(addr.ip()) || allow.permits(addr.ip()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv6Addr;

    fn v4(s: &str) -> IpAddr {
        s.parse().expect("v4 literal")
    }

    fn v6(s: &str) -> IpAddr {
        s.parse().expect("v6 literal")
    }

    fn sock(s: &str) -> SocketAddr {
        s.parse().expect("socket address")
    }

    fn allow(list: &str) -> ReachAllow {
        parse(Some("public"), Some(list))
            .expect("parsed")
            .expect("public")
    }

    /// A resolver that gives every name the same addresses.
    fn resolving_to(addrs: &'static [&'static str]) -> impl Fn(&str) -> Vec<IpAddr> {
        move |_| {
            addrs
                .iter()
                .map(|addr| addr.parse().expect("literal"))
                .collect()
        }
    }

    fn nothing(_: &str) -> Vec<IpAddr> {
        Vec::new()
    }

    /// `public` with `list`, its names resolved by `resolve`. Leaked: a policy
    /// borrows its list for the process's life, as the one read from the
    /// environment does.
    fn public_with(list: &str, resolve: impl Fn(&str) -> Vec<IpAddr>) -> NetPolicy {
        NetPolicy::Public(Box::leak(Box::new(allow(list).resolved(resolve))))
    }

    #[test]
    fn every_denied_v4_range_is_not_public() {
        for addr in [
            "0.0.0.0",
            "0.255.255.255",
            "10.0.0.1",
            "10.255.255.255",
            "100.64.0.1",
            "100.127.255.255",
            "127.0.0.1",
            "127.255.255.255",
            "169.254.0.1",
            "169.254.255.255",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.0.1",
            "192.168.255.255",
            "224.0.0.1",
            "239.255.255.255",
            "255.255.255.255",
        ] {
            assert!(!public_destination(v4(addr)), "{addr} should be refused");
        }
    }

    #[test]
    fn addresses_beside_the_denied_v4_ranges_are_public() {
        for addr in [
            "1.1.1.1",
            "8.8.8.8",
            "9.255.255.255",
            "11.0.0.0",
            "100.63.255.255",
            "100.128.0.0",
            "126.255.255.255",
            "128.0.0.1",
            "169.253.255.255",
            "169.255.0.0",
            "172.15.255.255",
            "172.32.0.0",
            "192.167.255.255",
            "192.169.0.0",
            "223.255.255.255",
            "240.0.0.1",
        ] {
            assert!(public_destination(v4(addr)), "{addr} should pass");
        }
    }

    #[test]
    fn every_denied_v6_range_is_not_public() {
        for addr in [
            "::", "::1", "fe80::1", "febf::1", "fc00::1", "fdff::1", "ff02::1", "ff00::",
        ] {
            assert!(!public_destination(v6(addr)), "{addr} should be refused");
        }
    }

    #[test]
    fn addresses_beside_the_denied_v6_ranges_are_public() {
        for addr in ["2001:db8::1", "2606:4700::1111", "fec0::1", "fbff::1"] {
            assert!(public_destination(v6(addr)), "{addr} should pass");
        }
    }

    #[test]
    fn a_v4_mapped_v6_address_is_judged_by_its_v4_half() {
        assert!(!public_destination(v6("::ffff:127.0.0.1")));
        assert!(!public_destination(v6("::ffff:10.0.0.1")));
        assert!(!public_destination(v6("::ffff:192.168.1.1")));
        assert!(public_destination(v6("::ffff:8.8.8.8")));
        assert!(public_destination(v6("::ffff:1.1.1.1")));
    }

    #[test]
    fn the_policy_parses_unset_empty_and_public_alone() {
        assert_eq!(parse(None, None).expect("unset"), None);
        assert_eq!(parse(Some(""), None).expect("empty"), None);
        assert_eq!(
            parse(Some("public"), None).expect("public"),
            Some(ReachAllow::default())
        );
    }

    #[test]
    fn a_policy_typo_is_an_error_naming_the_variable() {
        for bad in ["Public", "public ", " public", "PUBLIC", "strict", "none"] {
            let err = parse(Some(bad), None).expect_err("refused");
            let message = err.to_string();
            assert!(
                message.contains("FFRWD_NET_POLICY"),
                "error should name the variable, got: {message}"
            );
            assert!(
                message.contains("public"),
                "error should name the accepted value, got: {message}"
            );
        }
    }

    #[test]
    fn the_allow_list_parses_names_literals_and_prefixes_trimmed() {
        let parsed = allow(
            " ffrwd-pro-relay-egress.ffrwd.svc , 10.107.14.83,fd00::1 ,10.107.14.83/32, fd00::/8 ,",
        );
        assert_eq!(
            parsed.entries,
            vec![
                ReachEntry::Name("ffrwd-pro-relay-egress.ffrwd.svc".to_string()),
                ReachEntry::Addr(v4("10.107.14.83")),
                ReachEntry::Addr(v6("fd00::1")),
                ReachEntry::Cidr {
                    net: v4("10.107.14.83"),
                    bits: 32
                },
                ReachEntry::Cidr {
                    net: v6("fd00::"),
                    bits: 8
                },
            ]
        );
        assert!(parsed.resolved.is_empty(), "parsing resolves nothing");
    }

    #[test]
    fn an_empty_allow_list_is_no_exceptions() {
        assert_eq!(allow(""), ReachAllow::default());
        assert_eq!(allow(" , "), ReachAllow::default());
    }

    #[test]
    fn a_bad_allow_entry_is_an_error_naming_the_variable_and_the_entry() {
        for bad in [
            "10.107.14",
            "1.2.3.4.5",
            "relay.svc:4443",
            "10.0.0.0/33",
            "fd00::/129",
            "10.0.0.0/",
            "relay.svc/8",
            "-relay.svc",
            "relay..svc",
            "relay_svc",
            "http://relay.svc",
            "10.0.0.1 10.0.0.2",
        ] {
            let err = parse(Some("public"), Some(bad)).expect_err(bad);
            let message = err.to_string();
            assert!(
                message.contains("FFRWD_REACH_ALLOW"),
                "error should name the variable, got: {message}"
            );
            assert!(
                message.contains(bad),
                "error should quote the entry, got: {message}"
            );
        }
    }

    #[test]
    fn a_host_name_entry_may_be_short_qualified_or_dotted() {
        for name in [
            "relay",
            "relay.svc",
            "ffrwd-pro-relay-egress.ffrwd.svc",
            "relay.svc.",
        ] {
            assert_eq!(
                allow(name).entries,
                vec![ReachEntry::Name(name.to_string())],
                "{name} is a host name"
            );
        }
    }

    #[test]
    fn the_allow_list_is_ignored_when_unrestricted() {
        assert_eq!(parse(None, Some("not an entry!")).expect("ignored"), None);
        assert_eq!(parse(Some(""), Some("10.0.0.0/8")).expect("ignored"), None);
    }

    #[test]
    fn under_public_binds_pass_and_internal_destinations_are_refused() {
        let loopback: SocketAddr = "127.0.0.1:9".parse().expect("addr");
        let internet: SocketAddr = "1.1.1.1:9".parse().expect("addr");
        let policy = NetPolicy::Public(&NO_EXCEPTIONS);

        assert!(permit_socket_addr(policy, loopback, SocketAddrUse::UdpBind));
        assert!(permit_socket_addr(policy, loopback, SocketAddrUse::TcpBind));
        assert!(!permit_socket_addr(
            policy,
            loopback,
            SocketAddrUse::UdpSend
        ));
        assert!(!permit_socket_addr(
            policy,
            loopback,
            SocketAddrUse::TcpConnect
        ));
        assert!(permit_socket_addr(policy, internet, SocketAddrUse::UdpSend));
        assert!(permit_socket_addr(
            policy,
            internet,
            SocketAddrUse::TcpConnect
        ));
    }

    #[test]
    fn unrestricted_permits_every_use() {
        let loopback: SocketAddr = "127.0.0.1:9".parse().expect("addr");
        for reason in [
            SocketAddrUse::UdpBind,
            SocketAddrUse::UdpSend,
            SocketAddrUse::TcpConnect,
        ] {
            assert!(permit_socket_addr(
                NetPolicy::Unrestricted,
                loopback,
                reason
            ));
        }
    }

    #[test]
    fn a_v6_socket_destination_is_judged_like_its_address() {
        let mapped = SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), 9);
        assert!(!permit_socket_addr(
            NetPolicy::Public(&NO_EXCEPTIONS),
            mapped,
            SocketAddrUse::UdpSend
        ));
    }

    #[test]
    fn a_live_job_reaches_its_relays_and_nothing_else_private() {
        let policy = public_with(
            "10.107.14.83/32,ffrwd-pro-relay-ingress.ffrwd.svc",
            resolving_to(&["10.107.14.84"]),
        );
        let send = SocketAddrUse::UdpSend;
        assert!(permit_socket_addr(policy, sock("10.107.14.83:4443"), send));
        assert!(permit_socket_addr(policy, sock("10.107.14.84:4443"), send));
        assert!(!permit_socket_addr(policy, sock("10.0.0.9:4443"), send));
        assert!(!permit_socket_addr(policy, sock("10.107.14.85:4443"), send));
        assert!(!permit_socket_addr(policy, sock("127.0.0.1:4443"), send));
        assert!(permit_socket_addr(policy, sock("1.1.1.1:443"), send));
    }

    #[test]
    fn a_prefix_entry_holds_its_range_in_either_family() {
        let policy = public_with("10.107.14.0/24,fd00::/8", nothing);
        let connect = SocketAddrUse::TcpConnect;
        assert!(permit_socket_addr(policy, sock("10.107.14.1:1"), connect));
        assert!(permit_socket_addr(policy, sock("10.107.14.255:1"), connect));
        assert!(!permit_socket_addr(policy, sock("10.107.15.0:1"), connect));
        assert!(permit_socket_addr(policy, sock("[fd00::1]:1"), connect));
        assert!(permit_socket_addr(policy, sock("[fdff::1]:1"), connect));
        assert!(!permit_socket_addr(policy, sock("[fe80::1]:1"), connect));
        assert!(!permit_socket_addr(policy, sock("[fc00::1]:1"), connect));
    }

    #[test]
    fn an_address_entry_allows_a_send_and_leaves_binds_as_they_were() {
        let policy = public_with("192.168.1.20", nothing);
        let relay = sock("192.168.1.20:4443");
        let loopback = sock("127.0.0.1:0");
        assert!(permit_socket_addr(policy, relay, SocketAddrUse::UdpSend));
        assert!(permit_socket_addr(policy, loopback, SocketAddrUse::UdpBind));
        assert!(permit_socket_addr(
            policy,
            sock("0.0.0.0:0"),
            SocketAddrUse::TcpBind
        ));
        assert!(!permit_socket_addr(
            policy,
            loopback,
            SocketAddrUse::UdpSend
        ));
        assert!(!permit_socket_addr(
            policy,
            sock("192.168.1.21:4443"),
            SocketAddrUse::UdpSend
        ));
    }

    #[test]
    fn a_name_entry_allows_what_it_resolved_to() {
        let policy = public_with(
            "relay.ffrwd.svc",
            resolving_to(&["10.107.14.83", "fd00::83"]),
        );
        let send = SocketAddrUse::UdpSend;
        assert!(permit_socket_addr(policy, sock("10.107.14.83:4443"), send));
        assert!(permit_socket_addr(policy, sock("[fd00::83]:4443"), send));
        assert!(!permit_socket_addr(policy, sock("10.107.14.84:4443"), send));
    }

    #[test]
    fn a_name_that_does_not_resolve_allows_nothing() {
        let policy = public_with("relay.ffrwd.svc", nothing);
        assert!(!permit_socket_addr(
            policy,
            sock("10.107.14.83:4443"),
            SocketAddrUse::UdpSend
        ));
        let NetPolicy::Public(allow) = policy else {
            panic!("public");
        };
        assert!(
            !allow.is_empty(),
            "the list is set even when nothing resolved"
        );
    }

    #[test]
    fn a_v4_mapped_destination_is_matched_by_its_v4_half() {
        let policy = public_with("10.107.14.83,10.107.15.0/24", nothing);
        let send = SocketAddrUse::UdpSend;
        assert!(permit_socket_addr(
            policy,
            sock("[::ffff:10.107.14.83]:4443"),
            send
        ));
        assert!(permit_socket_addr(
            policy,
            sock("[::ffff:10.107.15.9]:4443"),
            send
        ));
        assert!(!permit_socket_addr(
            policy,
            sock("[::ffff:10.107.16.9]:4443"),
            send
        ));
    }

    #[test]
    fn the_http_path_dials_every_address_of_an_allowed_name() {
        let allow = allow("relay.ffrwd.svc").resolved(nothing);
        let resolved = vec![sock("10.107.14.83:443"), sock("1.1.1.1:443")];
        assert_eq!(
            dialable(&allow, "relay.ffrwd.svc", resolved.clone()),
            resolved
        );
        assert_eq!(
            dialable(&allow, "RELAY.ffrwd.svc.", resolved.clone()),
            resolved
        );
        assert_eq!(
            dialable(&allow, "other.ffrwd.svc", resolved),
            vec![sock("1.1.1.1:443")]
        );
    }

    #[test]
    fn the_http_path_dials_an_allowed_literal_and_filters_the_rest() {
        let allow = allow("10.107.14.83,fd00::/8");
        assert_eq!(
            dialable(&allow, "10.107.14.83", vec![sock("10.107.14.83:443")]),
            vec![sock("10.107.14.83:443")]
        );
        assert_eq!(
            dialable(&allow, "fd00::9", vec![sock("[fd00::9]:443")]),
            vec![sock("[fd00::9]:443")]
        );
        assert!(dialable(&allow, "10.0.0.9", vec![sock("10.0.0.9:443")]).is_empty());
        assert_eq!(
            dialable(
                &allow,
                "mixed.example",
                vec![
                    sock("10.0.0.9:443"),
                    sock("8.8.8.8:443"),
                    sock("10.107.14.83:443")
                ]
            ),
            vec![sock("8.8.8.8:443"), sock("10.107.14.83:443")]
        );
    }

    #[test]
    fn the_refusal_names_the_list_only_when_it_is_set() {
        assert_eq!(NO_EXCEPTIONS.allowed_scope(), "public destinations only");
        assert_eq!(
            allow("10.0.0.0/8").allowed_scope(),
            "public destinations and FFRWD_REACH_ALLOW only"
        );
    }
}
