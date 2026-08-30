"""ffprobe wrapper for ffrwd.

`probe()` NEVER raises. Every failure mode -- a missing file, ffprobe absent
from PATH and its provisioner both, a nonzero ffprobe exit, a timeout, or
unparseable JSON -- returns `None`, and callers fall back to symbolic
lowering. This module depends only on `ffrwd.ir` (`StreamType`),
`ffrwd.binaries` (locating ffprobe) and the stdlib; it must never import
anything else from the package.

A local-path existence check runs BEFORE `binaries.ffprobe_path()` is even
consulted: a missing file is `None` with no subprocess AND no
provider lookup, which matters because the provider's first call may trigger
a ~95MB download -- paying that once per compile for an input that does not
even exist would be its own footgun. An input whose options force the demuxer
skips that check -- its spec names a device or a graph, not a file.

Results are memoized per `(realpath, mtime_ns, size, input flags)` so a
compile that probes the same input multiple times only shells out once, while
one path read two ways -- different `input()` options -- stays two probes;
`clear_cache()` resets the memo for tests. A failed probe is memoized on the
same key, as a `ProbeFailure` a caller can read back with `probe_failure()`
-- the detail behind a `None` for a path that DOES exist, so a rejection
downstream can say what actually went wrong instead of guessing "not found".
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from ffrwd import binaries
from ffrwd.ir import StreamType

_TIMEOUT_SECONDS = 5.0
# Remote specs fetch a manifest and often an init segment per stream before
# ffprobe can report anything; 5s flakes on real networks, so they get more.
_REMOTE_TIMEOUT_SECONDS = 15.0

# ffprobe's own name for the WebVTT demuxer, and the document's first word.
WEBVTT_FORMAT = "webvtt"
_WEBVTT_MAGIC = "WEBVTT"
_CUE_ARROW = "-->"
# Blocks that are not cues: comments, styling, regions.
_NOT_CUE_BLOCKS = ("NOTE", "STYLE", "REGION", _WEBVTT_MAGIC)
# WebVTT writes its payload with HTML's character references. `&amp;` is read
# LAST so that an escaped ampersand does not turn a following name into one.
_WEBVTT_UNESCAPES = (("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"))


@dataclass(frozen=True)
class StreamMeta:
    """Audio rows carry channels/channel_layout/sample_rate, video rows carry
    width/height/fps/color_transfer; codec/bitrate/duration
    are common to both. Every field is opportunistic -- absent or wrong-typed
    in ffprobe's JSON means None, never an exception (see
    `_int_opt`/`_float_opt`/`_str_opt`), and a defaulted field means exactly
    that: not supplied.
    """

    type: StreamType
    index: int  # per-type, 0-based (0:a:<index>)
    metadata: dict[str, str]  # the stream's tags in full, keys lowercased
    width: int | None
    height: int | None
    fps: str | None  # e.g. "30000/1001", verbatim from ffprobe avg_frame_rate
    sample_rate: int | None
    codec: str | None = None  # ffprobe codec_name, verbatim
    channels: int | None = None  # audio only
    channel_layout: str | None = None  # audio only, e.g. "stereo"
    bitrate: int | None = None  # ffprobe bit_rate, as int
    duration: float | None = None  # per-stream duration in seconds
    color_transfer: str | None = None  # video only; the HDR discriminator
    # ffprobe's `disposition` object as booleans, keys lowercased. The
    # flag map `<row>.disposition.<key>` reads.
    disposition: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class ChapterMeta:
    """One chapter, from ``ffprobe -show_chapters``.

    `index` is 1-based, in ffprobe's own order -- the raw ``id`` field is
    container-specific (a remuxed mkv starts it at 1, not 0) and not reused
    here. `start_t`/`end_t` are seconds, read from ``start_time``/``end_time``
    (already decimal strings ffprobe derives from the chapter's own time
    base); `title` comes from the chapter's tags, same convention as a
    stream's. Every field is opportunistic, like :class:`StreamMeta`.
    """

    index: int
    start_t: float | None
    end_t: float | None
    title: str | None


@dataclass(frozen=True)
class AttachmentMeta:
    """One attached file, from a stream ffprobe reports as ``attachment``.

    An attachment is not a stream ffrwd maps -- ffmpeg attaches a file by
    path -- so these are kept OUT of :attr:`ProbeResult.streams`, and the
    per-type indices there keep counting only the streams that are mapped.
    `index` is the attachment's place in the file, 1-based;
    `filename`/`mimetype` come from its tags, and are None when absent.
    """

    index: int
    filename: str | None
    mimetype: str | None


@dataclass(frozen=True)
class CueMeta:
    """One WebVTT cue, read from the document itself.

    ffprobe does not enumerate cues, so these come from parsing the ``.vtt``
    file: see :func:`parse_webvtt`. `index` is the cue's place in the
    document, 1-based; `start_t`/`end_t` are seconds; `text` is the payload
    with its lines joined by newlines and WebVTT's escapes read back.
    """

    index: int
    text: str
    start_t: float
    end_t: float


@dataclass(frozen=True)
class ProbeResult:
    streams: list[StreamMeta]  # file order
    duration: float | None = None  # container-level, from -show_format
    chapters: list[ChapterMeta] = field(default_factory=list)
    # ffprobe's own name for the demuxer that read the file ("webvtt",
    # "matroska,webm", ...), verbatim. It is what says a file IS a WebVTT
    # document and so has cues to read.
    format_name: str | None = None
    # The document's cues, for a WebVTT input and nothing else.
    cues: list[CueMeta] = field(default_factory=list)
    # The files riding inside the container, in ffprobe's order. A file with
    # none has an empty list.
    attachments: list[AttachmentMeta] = field(default_factory=list)
    # Container-level tags from -show_format, keys lowercased, values verbatim.
    # The WHOLE tag dict, not a whitelist: which keys a query may read is
    # decided where they resolve, not here.
    tags: dict[str, str] = field(default_factory=dict)

    def by_type(self, t: StreamType) -> list[StreamMeta]:
        return [s for s in self.streams if s.type == t]


@dataclass(frozen=True)
class ProbeFailure:
    """Why `probe()` returned None for an input whose path exists and is readable.

    A companion to the plain `None` `probe()` returns for EVERY failure mode,
    told only for the one that a caller can honestly report on: the path
    stat'd fine, so whatever went wrong belongs to the probe attempt itself,
    not to the file's existence. `stderr` is ffprobe's own last line of
    diagnostic output on a nonzero exit; None when ffprobe never got that
    far (its binary missing, a timeout, or output that was not the JSON
    shape expected).
    """

    stderr: str | None


# (realpath, mtime_ns, size, input flags)
_CacheKey = tuple[str, int, int, tuple[str, ...]]
_cache: dict[_CacheKey, ProbeResult] = {}
_failure_cache: dict[_CacheKey, ProbeFailure] = {}


def clear_cache() -> None:
    """Clear the probe() memoization cache. For tests."""
    _cache.clear()
    _failure_cache.clear()


def is_url(spec: str) -> bool:
    """True when `spec` names a protocol (``scheme://...``) rather than a file.

    The one shared rule for udp, rtmp, srt, http and the rest: ffmpeg is the
    authority on its own protocols, so a "://" spec is handed over verbatim --
    never existence-checked, never treated as a path.
    """
    return "://" in spec


def probe(
    path: str,
    args: Sequence[str] = (),
    *,
    forced_format: bool = False,
) -> ProbeResult | None:
    """Probe a media input with ffprobe.

    Returns None -- never raises -- when: the file does not exist, ffprobe
    is not on PATH or via its provisioner, ffprobe exits nonzero or times
    out, or its output is not the JSON shape we expect.
    :func:`probe_failure` distinguishes those cases after the fact, for a
    path that DOES exist -- there is nothing more to say about one that
    doesn't.

    `args` are the input's own options, already rendered to argv by
    :func:`ffrwd.inputs.render_options`, and are passed to ffprobe ahead of
    the spec -- the same flags in the same order the decode gets, so both
    read the input the same way. They are part of the cache key: one path
    read two ways is two probes.

    `forced_format` says the options name the demuxer (``format => ...``),
    which makes the spec ITS to interpret -- a device name, a lavfi graph --
    so no existence check runs and the spec goes to ffprobe verbatim. Such a
    spec has no mtime to key on, like a URL, so its cache entry is
    per-process "what this said when we asked". Opening a device is slower
    than reading a file, so it gets the longer timeout.

    A spec containing "://" is handed to ffprobe VERBATIM too -- ffprobe is
    the authority on its own protocols, so a remote input probes over the
    network and an unsupported scheme fails into the same permissive None.
    """
    flags = tuple(args)
    if forced_format or is_url(path):
        return _cached_ffprobe(path, (path, -1, -1, flags), _REMOTE_TIMEOUT_SECONDS, flags)

    try:
        if not os.path.isfile(path):
            return None
        real = os.path.realpath(path)
        st = os.stat(real)
    except OSError:
        return None

    return _cached_ffprobe(
        real, (real, st.st_mtime_ns, st.st_size, flags), _TIMEOUT_SECONDS, flags
    )


def _cached_ffprobe(
    spec: str, cache_key: _CacheKey, timeout: float, flags: tuple[str, ...] = ()
) -> ProbeResult | None:
    """One memoized ffprobe invocation over `spec` (a path, URL or device).

    Every failure branch records why in `_failure_cache`, keyed the same as
    a success -- :func:`probe_failure` is the other half of this cache, read
    by a caller that already knows `probe()` came back None for `cache_key`
    and wants to say something truer than "unreadable" about a path that
    does exist.
    """
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    ffprobe = binaries.ffprobe_path()
    if ffprobe is None:
        _failure_cache[cache_key] = ProbeFailure(stderr=None)
        return None

    argv = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        "-show_chapters",
        *flags,
        spec,
    ]
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        _failure_cache[cache_key] = ProbeFailure(stderr=None)
        return None

    if result.returncode != 0:
        _failure_cache[cache_key] = ProbeFailure(stderr=_last_line(result.stderr))
        return None

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        _failure_cache[cache_key] = ProbeFailure(stderr=None)
        return None

    parsed = _parse_streams(data)
    if parsed is None:
        _failure_cache[cache_key] = ProbeFailure(stderr=None)
        return None

    parsed = _with_cues(parsed, spec)
    _cache[cache_key] = parsed
    return parsed


def _last_line(text: str) -> str | None:
    """The last non-blank line of `text`, or None when there isn't one.

    ffprobe's ``-v error`` diagnostics are one message per line; the last
    one is usually the actual cause, with any earlier lines being context
    (an opened-but-then-rejected option, a probed-then-abandoned format).
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else None


def probe_failure(
    path: str,
    args: Sequence[str] = (),
    *,
    forced_format: bool = False,
) -> ProbeFailure | None:
    """Why the last `probe()` of `path` under these `args` came back None.

    None both when that probe actually succeeded and when there is nothing
    more specific to say than "unreadable": a forced-format spec or a URL
    names no local file to stat, and a path that fails the existence check
    itself never reached ffprobe at all. Same existence check as `probe()`,
    so the two agree on which paths this has an answer for.
    """
    flags = tuple(args)
    if forced_format or is_url(path):
        return None
    try:
        if not os.path.isfile(path):
            return None
        real = os.path.realpath(path)
        st = os.stat(real)
    except OSError:
        return None
    return _failure_cache.get((real, st.st_mtime_ns, st.st_size, flags))


def _with_cues(parsed: ProbeResult, spec: str) -> ProbeResult:
    """`parsed` plus the cues of a WebVTT document, and nothing else.

    ffprobe reports a WebVTT file's ONE subtitle stream and stops there -- it
    never lists cues -- so the document is read a second time, as text, by
    :func:`parse_webvtt`. Only a local file ffprobe already identified as
    ``webvtt`` is read: a remote spec is not fetched, and a container that
    merely CARRIES a webvtt track is not demuxed, so neither has cues here.
    Unreadable or undecodable text leaves the list empty, like every other
    failure in this module.
    """
    if parsed.format_name != WEBVTT_FORMAT or is_url(spec):
        return parsed
    try:
        with open(spec, encoding="utf-8-sig") as handle:
            text = handle.read()
    except (OSError, UnicodeDecodeError):
        return parsed
    return replace(parsed, cues=parse_webvtt(text))


def _parse_streams(data: object) -> ProbeResult | None:
    try:
        if not isinstance(data, dict):
            return None
        raw_streams = data["streams"]
        if not isinstance(raw_streams, list):
            return None

        streams: list[StreamMeta] = []
        attachments: list[AttachmentMeta] = []
        video_idx = 0
        audio_idx = 0
        subtitle_idx = 0
        data_idx = 0
        for raw in raw_streams:
            if not isinstance(raw, dict):
                return None
            codec_type = raw["codec_type"]

            codec = _str_opt(raw, "codec_name")
            bitrate = _int_opt(raw, "bit_rate")
            duration = _float_opt(raw, "duration")
            flags = _dispositions(raw)

            if codec_type == "video":
                metadata = _tags(raw)
                streams.append(
                    StreamMeta(
                        type="video",
                        index=video_idx,
                        metadata=metadata,
                        width=int(raw["width"]) if "width" in raw else None,
                        height=int(raw["height"]) if "height" in raw else None,
                        fps=str(raw["avg_frame_rate"]) if "avg_frame_rate" in raw else None,
                        sample_rate=None,
                        codec=codec,
                        channels=None,
                        channel_layout=None,
                        bitrate=bitrate,
                        duration=duration,
                        color_transfer=_str_opt(raw, "color_transfer"),
                        disposition=flags,
                    )
                )
                video_idx += 1
            elif codec_type == "audio":
                metadata = _tags(raw)
                streams.append(
                    StreamMeta(
                        type="audio",
                        index=audio_idx,
                        metadata=metadata,
                        width=None,
                        height=None,
                        fps=None,
                        sample_rate=int(raw["sample_rate"]) if "sample_rate" in raw else None,
                        codec=codec,
                        channels=_int_opt(raw, "channels"),
                        channel_layout=_str_opt(raw, "channel_layout"),
                        bitrate=bitrate,
                        duration=duration,
                        color_transfer=None,
                        disposition=flags,
                    )
                )
                audio_idx += 1
            elif codec_type == "subtitle":
                metadata = _tags(raw)
                streams.append(
                    StreamMeta(
                        type="subtitle",
                        index=subtitle_idx,
                        metadata=metadata,
                        width=None,
                        height=None,
                        fps=None,
                        sample_rate=None,
                        codec=codec,
                        channels=None,
                        channel_layout=None,
                        bitrate=bitrate,
                        duration=duration,
                        color_transfer=None,
                        disposition=flags,
                    )
                )
                subtitle_idx += 1
            elif codec_type == "data":
                metadata = _tags(raw)
                streams.append(
                    StreamMeta(
                        type="data",
                        index=data_idx,
                        metadata=metadata,
                        width=None,
                        height=None,
                        fps=None,
                        sample_rate=None,
                        codec=codec,
                        channels=None,
                        channel_layout=None,
                        bitrate=bitrate,
                        duration=duration,
                        color_transfer=None,
                        disposition=flags,
                    )
                )
                data_idx += 1
            elif codec_type == "attachment":
                tags = _tags(raw)
                attachments.append(
                    AttachmentMeta(
                        index=len(attachments) + 1,
                        filename=tags.get("filename"),
                        mimetype=tags.get("mimetype"),
                    )
                )
            # other codec_type values are ignored.

        container_duration = None
        container_tags: dict[str, str] = {}
        format_name: str | None = None
        raw_format = data.get("format")
        if isinstance(raw_format, dict):
            container_duration = _float_opt(raw_format, "duration")
            container_tags = _tags(raw_format)
            format_name = _str_opt(raw_format, "format_name")

        chapters = _parse_chapters(data.get("chapters"))

        return ProbeResult(
            streams=streams,
            duration=container_duration,
            chapters=chapters,
            format_name=format_name,
            tags=container_tags,
            attachments=attachments,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _parse_chapters(raw_chapters: object) -> list[ChapterMeta]:
    """``data["chapters"]`` as a list of :class:`ChapterMeta`, in ffprobe's order.

    Permissive like everything else here: a malformed chapter entry is
    dropped rather than failing the whole probe -- the file's streams are
    still good even when one chapter's tags are not.
    """
    if not isinstance(raw_chapters, list):
        return []
    chapters: list[ChapterMeta] = []
    for index, raw in enumerate(raw_chapters):
        if not isinstance(raw, dict):
            continue
        tags = raw.get("tags", {})
        title = str(tags["title"]) if isinstance(tags, dict) and "title" in tags else None
        chapters.append(
            ChapterMeta(
                index=index + 1,
                start_t=_float_opt(raw, "start_time"),
                end_t=_float_opt(raw, "end_time"),
                title=title,
            )
        )
    return chapters


def parse_webvtt(text: str) -> list[CueMeta]:
    """Every cue of a WebVTT document, in document order, numbered from 1.

    Permissive like the rest of this module: a block whose timing line does
    not parse is skipped rather than failing the read, and so are the
    ``WEBVTT`` header and the ``NOTE``/``STYLE``/``REGION`` blocks. Cue
    settings after the end timestamp (``line:0``, ``align:start``) are read
    and dropped -- they position the text, and ffrwd exposes the text.
    """
    cues: list[CueMeta] = []
    for block in _blocks(text):
        cue = _parse_cue(block, len(cues) + 1)
        if cue is not None:
            cues.append(cue)
    return cues


def _blocks(text: str) -> list[list[str]]:
    """A document's blocks: the runs of non-blank lines a blank line separates."""
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip():
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _parse_cue(block: list[str], index: int) -> CueMeta | None:
    """One block as a cue, or None when it is not one.

    The timing line is the first or the second line -- a cue may carry an
    identifier line ahead of it -- and everything after it is the payload.
    """
    if block[0].split()[0] in _NOT_CUE_BLOCKS:
        return None
    position = next((n for n in (0, 1) if n < len(block) and _CUE_ARROW in block[n]), None)
    if position is None:
        return None
    left, _, right = block[position].partition(_CUE_ARROW)
    start = _cue_time(left.strip())
    settings = right.split()
    end = _cue_time(settings[0]) if settings else None
    if start is None or end is None:
        return None
    return CueMeta(
        index=index,
        text=_unescaped("\n".join(block[position + 1 :])),
        start_t=start,
        end_t=end,
    )


def _cue_time(value: str) -> float | None:
    """``HH:MM:SS.mmm`` or ``MM:SS.mmm`` as seconds, or None if it is neither."""
    parts = value.split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        seconds = float(parts[-1])
        minutes = int(parts[-2])
        hours = int(parts[-3]) if len(parts) == 3 else 0
    except ValueError:
        return None
    if min(hours, minutes, seconds) < 0:
        return None
    return hours * 3600 + minutes * 60 + seconds


def _unescaped(payload: str) -> str:
    """A cue payload with WebVTT's character references read back."""
    for escape, character in _WEBVTT_UNESCAPES:
        payload = payload.replace(escape, character)
    return payload


def _int_opt(raw: dict[str, object], key: str) -> int | None:
    """`raw[key]` as `int`, or None if absent, None-valued, or unparseable.

    Never raises: a per-field escape hatch so one malformed value nulls only
    that column instead of failing the whole probe (unlike the outer
    try/except in `_parse_streams`, which nulls the entire result).
    """
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_opt(raw: dict[str, object], key: str) -> float | None:
    """`raw[key]` as `float`, or None if absent, None-valued, or unparseable."""
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _str_opt(raw: dict[str, object], key: str) -> str | None:
    """`raw[key]` as `str`, or None if absent or None-valued."""
    if key not in raw or raw[key] is None:
        return None
    return str(raw[key])


def _dispositions(raw: dict[str, object]) -> dict[str, bool]:
    """A ``disposition`` object as booleans, keys lowercased.

    ffprobe prints 1/0 per flag; a value that is neither is dropped rather
    than guessed, the same way every other field here nulls itself instead of
    failing the whole probe.
    """
    flags = raw.get("disposition")
    if not isinstance(flags, dict):
        return {}
    return {
        str(key).lower(): bool(value)
        for key, value in flags.items()
        if isinstance(value, (bool, int))
    }


def _tags(raw: dict[str, object]) -> dict[str, str]:
    """A ``tags`` object in full, keys lowercased (muxers vary the case).

    One function for both levels: a stream's tags and the container's are the
    same free-form map, and which keys a query may read is decided where they
    resolve, not here.
    """
    tags = raw.get("tags")
    if not isinstance(tags, dict):
        return {}
    return {str(key).lower(): str(value) for key, value in tags.items()}
