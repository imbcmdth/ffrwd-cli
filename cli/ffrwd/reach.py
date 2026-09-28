"""The addresses a hosted live run may not reach.

A hosted live job runs with the network open: its readers dial a relay or an
origin, its writers publish to one, and its nodes reach each other over the
provider's private network. What it must not reach is that private network's
other machines, the provider's metadata service, or its own loopback: a query
naming one is refused before anything starts.

:func:`private_destinations` reads the addresses a compiled query names off
its graphs and its sidecars' arguments, and answers each one that names such
a place. It reads what is written, a literal address or a name that can only
mean a private host; a public name that resolves to a private address is the
network's to stop, not this check's. A listening input's address (``listen``,
SRT in listener mode, any UDP or RTP input) is where its own socket binds, not
a place it reaches, and is left alone.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Iterable, Sequence
from urllib.parse import parse_qs, urlsplit

from .ir import Graph
from .processes import ProcessPlan, SidecarProcess

__all__ = ["private_destinations", "private_reason"]

# Names that only ever mean a host on this machine or its private network.
_PRIVATE_NAMES = ("localhost",)
_PRIVATE_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".home.arpa")


def private_reason(url: str) -> str | None:
    """Why `url` names a private place, or None when it does not (or names no
    host at all: a file, a pipe, a data document)."""
    if "://" not in url:
        return None
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return None
    if not host:
        return None
    host = host.rstrip(".").lower()
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if host in _PRIVATE_NAMES or host.endswith(_PRIVATE_SUFFIXES):
            return f"'{host}' names a private host"
        return None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    for test, what in (
        (address.is_unspecified, "an unspecified address, which reaches this machine"),
        (address.is_loopback, "a loopback address"),
        (address.is_link_local, "a link-local address"),
        (address.is_private, "a private address"),
        (address.is_multicast, "a multicast address"),
        (address.is_reserved, "a reserved address"),
    ):
        if test:
            return f"{host} is {what}"
    return None


def private_destinations(
    graphs: Iterable[Graph], plan: ProcessPlan | None = None
) -> list[tuple[str, str]]:
    """Each ``(url, reason)`` a compiled query names that reaches a private
    place: an input it dials, a destination it writes, or a URL a sidecar is
    given as an argument. In written order, without repeats."""
    found: dict[str, str] = {}

    def check(url: str) -> None:
        reason = private_reason(url)
        if reason is not None and url not in found:
            found[url] = reason

    for graph in graphs:
        listening = _listening(graph)
        for index, path in enumerate(graph.input_paths):
            if index not in listening:
                check(path)
        for unit in graph.sinks:
            if unit.path:
                check(unit.path)
    if plan is not None:
        for process in plan.processes:
            if isinstance(process, SidecarProcess):
                for value in _strings(process.args.values()):
                    check(value)
    return list(found.items())


def _listening(graph: Graph) -> set[int]:
    """The input indexes of `graph` that listen rather than dial: an input
    given ``listen``, an rtmp one asking ``?listen=1``, SRT in listener mode,
    and every UDP or RTP input, whose address is the one it binds (or the
    group it joins)."""
    found: set[int] = set()
    for alias, index in graph.sources.items():
        options = graph.input_options.get(alias, {})
        if options.get("listen") in (True, "true", "1", 1):
            found.add(index)
            continue
        parts = urlsplit(graph.input_paths[index])
        scheme = parts.scheme.lower()
        query = parse_qs(parts.query)
        if (
            scheme in ("udp", "rtp")
            or (scheme == "srt" and "listener" in query.get("mode", []))
            or (scheme.startswith("rtmp") and "1" in query.get("listen", []))
        ):
            found.add(index)
    return found


def _strings(values: Iterable[object]) -> Sequence[str]:
    """Every string among `values`, looking inside lists and mappings."""
    out: list[str] = []
    for value in values:
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, dict):
            out += _strings(value.values())
        elif isinstance(value, list | tuple):
            out += _strings(value)
    return out
