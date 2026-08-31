//! Outbound network policy, read from `FFRWD_NET_POLICY`. Unset leaves the
//! two grants (`-net`, `-http`) unrestricted; `public` refuses destinations
//! in the private, loopback, link-local, carrier-NAT, multicast and
//! broadcast ranges. Local binds are never restricted.
//!
//! For wasi:sockets the check runs per address use. For wasi:http the
//! request path resolves the authority itself and connects only to public
//! addresses, so a hostname is judged by what it resolves to, not by its
//! spelling.

use std::future::Future;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
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
    /// Public destinations only.
    Public,
}

/// Parses one `FFRWD_NET_POLICY` value. Exact match: a typo is an error,
/// never an open network.
pub fn parse(value: Option<&str>) -> Result<NetPolicy> {
    match value {
        None | Some("") => Ok(NetPolicy::Unrestricted),
        Some("public") => Ok(NetPolicy::Public),
        Some(other) => bail!(
            "FFRWD_NET_POLICY is {other:?}: the only accepted value is \"public\"; \
             unset it to leave the network unrestricted"
        ),
    }
}

/// The process's policy, read from the environment once.
pub fn net_policy() -> Result<NetPolicy> {
    static POLICY: OnceLock<std::result::Result<NetPolicy, String>> = OnceLock::new();
    POLICY
        .get_or_init(|| {
            let read = match std::env::var("FFRWD_NET_POLICY") {
                Ok(value) => return parse(Some(&value)).map_err(|e| e.to_string()),
                Err(read) => read,
            };
            match read {
                std::env::VarError::NotPresent => Ok(NetPolicy::Unrestricted),
                std::env::VarError::NotUnicode(_) => Err(
                    "FFRWD_NET_POLICY is not unicode: the only accepted value is \"public\"; \
                     unset it to leave the network unrestricted"
                        .to_string(),
                ),
            }
        })
        .clone()
        .map_err(|message| anyhow!(message))
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
/// Binds and inbound addresses pass; only where traffic goes is policed.
pub fn permit_socket_addr(policy: NetPolicy, addr: SocketAddr, reason: SocketAddrUse) -> bool {
    match policy {
        NetPolicy::Unrestricted => true,
        NetPolicy::Public => match reason {
            SocketAddrUse::TcpBind
            | SocketAddrUse::TcpListen
            | SocketAddrUse::TcpAccept
            | SocketAddrUse::UdpBind
            | SocketAddrUse::UdpReceive => true,
            SocketAddrUse::TcpConnect | SocketAddrUse::UdpSend => public_destination(addr.ip()),
        },
    }
}

/// The address check installed under `public`: permits or refuses, and says
/// so on stderr when refusing.
pub(crate) fn check_socket_addr(addr: SocketAddr, reason: SocketAddrUse) -> bool {
    let allowed = permit_socket_addr(NetPolicy::Public, addr, reason);
    if !allowed {
        eprintln!(
            "ffrwd-wasm: refused {addr}: FFRWD_NET_POLICY=public allows public destinations only"
        );
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
            NetPolicy::Public => Box::new(send_request_public(request, options)),
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
/// the authority is resolved here and only public addresses are connected
/// to. Resolving and connecting in one place is what closes the gap between
/// checking a name and dialing it.
async fn send_request_public(
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
    let public: Vec<SocketAddr> = candidates
        .into_iter()
        .filter(|addr| public_destination(addr.ip()))
        .collect();
    if public.is_empty() {
        eprintln!(
            "ffrwd-wasm: refused http request to {host}:{port}: \
             FFRWD_NET_POLICY=public allows public destinations only"
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
    for addr in public {
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
        assert_eq!(parse(None).expect("unset"), NetPolicy::Unrestricted);
        assert_eq!(parse(Some("")).expect("empty"), NetPolicy::Unrestricted);
        assert_eq!(parse(Some("public")).expect("public"), NetPolicy::Public);
    }

    #[test]
    fn a_policy_typo_is_an_error_naming_the_variable() {
        for bad in ["Public", "public ", " public", "PUBLIC", "strict", "none"] {
            let err = parse(Some(bad)).expect_err("refused");
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
    fn under_public_binds_pass_and_internal_destinations_are_refused() {
        let loopback: SocketAddr = "127.0.0.1:9".parse().expect("addr");
        let internet: SocketAddr = "1.1.1.1:9".parse().expect("addr");

        assert!(permit_socket_addr(
            NetPolicy::Public,
            loopback,
            SocketAddrUse::UdpBind
        ));
        assert!(permit_socket_addr(
            NetPolicy::Public,
            loopback,
            SocketAddrUse::TcpBind
        ));
        assert!(!permit_socket_addr(
            NetPolicy::Public,
            loopback,
            SocketAddrUse::UdpSend
        ));
        assert!(!permit_socket_addr(
            NetPolicy::Public,
            loopback,
            SocketAddrUse::TcpConnect
        ));
        assert!(permit_socket_addr(
            NetPolicy::Public,
            internet,
            SocketAddrUse::UdpSend
        ));
        assert!(permit_socket_addr(
            NetPolicy::Public,
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
            NetPolicy::Public,
            mapped,
            SocketAddrUse::UdpSend
        ));
    }
}
