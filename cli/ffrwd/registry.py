"""ffmpeg filter registry for ffrwd.

Introspects the ffmpeg binary on PATH for the FULL set of filters it
supports (~460-560 depending on build), beyond the curated stdlib. Two
ffmpeg CLI outputs drive this:

  `ffmpeg -hide_banner -filters` -- one line per filter: a 3-character flag
  column (T/. S/. C/.), the filter name, a pad spec (`V->V`, `AA->A`, `N`
  for dynamic pad count, `|` for source/sink), and a one-line description.
  This is the ONLY source for pad signatures; -help does not restate them
  in a machine-parseable way for sources/sinks.

  `ffmpeg -hide_banner -help filter=<name>` -- per-option lines of the form
  `   <name>   <type>   <flags>   <description> (from A to B) (default D)`
  at a fixed 3-space indent, with enum constants as `<name> <value> <flags>
  <description>` lines indented 5 spaces directly below their parent
  option.

NEVER raises: any subprocess failure, timeout, missing ffmpeg, or unparseable
output degrades permissively -- an unparseable option becomes type str rather
than dropping the whole filter. Depends only on `ffrwd.ir` (for
`StreamType`) and the stdlib.

MEASURED format quirks this parser is built around. All figures below come
from a full scan of `ffmpeg version 7.1-full_build-www.gyan.dev` (captured
2026-08); tests/test_registry.py holds the captured fixtures.

  - The `-filters` legend (`Filters:` + 7 `X = ...` lines) is never skipped
    by line count -- the per-line regex requires the 3-char flag column at
    column 1, which legend lines (two leading spaces) never match. Resilient
    to legend wording changes across versions.
  - Pad specs seen across ~560 filters: `A->A`, `AA->A`, `V->V`, `VV->V`,
    `VVV->V`, `VVVV->V`, `A->V`. `VV->A` was never observed; mixed in/out
    letters parse verbatim per character regardless. Multi-output (`VV->VV`:
    `feedback`, `scale2ref`), a dynamic pad count on the OUTPUT side (`split`:
    `V->N`, `concat`/`streamselect`/`astreamselect`: `N->N`) and source/sink
    `|` (`testsrc`: `|->V`, `anullsink`: `A->|`) are excluded by the pad scope
    check. A dynamic pad count on the INPUT side alone (`N->V`: `hstack`,
    `N->A`: `amix`, ~31 filters on ffmpeg 9.0.1) is INCLUDED, marked
    `DynamicFilter.n_input`, with the input pads' stream type read off the
    single output letter -- every one observed takes pads of that same kind.
    Every zero-input filter observed uses `|` as its input character, never
    an empty string -- the "zero inputs" exclusion is defensive, not
    something seen in practice.
  - The `-help filter=X` AVOptions header does NOT always read
    `"X AVOptions:"`: filters sharing an implementation share a header
    (`split` -> `"(a)split AVOptions:"`, `acompressor` ->
    `"acompressor/sidechaincompress AVOptions:"`), 51 of ~460 included
    filters mismatched. The parser therefore never matches header text
    against the filter name; it takes the FIRST line containing
    "AVOptions:", unconditionally.
  - Some help outputs have MULTIPLE "AVOptions:" sections separated by a
    blank line: `scale` has its own, then `SWScaler AVOptions:` (whose
    option names are literally `-`-prefixed, e.g. `-sws_flags`, at 2-space
    indent), then `framesync AVOptions:`; `overlay` has its own then
    `framesync`. Only the FIRST block is parsed -- later sections belong to
    a different AVClass, not this filter.
  - Filters with no options (e.g. `anullsink`) have no "AVOptions:" line at
    all; that degrades to an empty options dict, not an error.
  - Short/long option aliases appear as separate AVOption lines with
    IDENTICAL description text (`subtitles`: `filename`/`f`; `testsrc`:
    `size`/`s`, `rate`/`r`; `scale`: `w`/`width`, `h`/`height`). The LONGER
    name is the one to keep: 456 long-first vs 29 short-first vs 17
    equal-length duplicate-doc groups. File ORDER is not a reliable signal
    on its own -- `scale` lists `w` before `width` but `size` before `s`.
    So: dedup by identical doc text, keep the longest name, break length
    ties by first occurrence -- but ONLY within a run of CONSECUTIVE lines
    sharing that doc text.

    Adjacency is ffmpeg's own rule, not a heuristic. An alias pair is two
    AVOption entries at the same struct OFFSET, and libavfilter's
    `process_options` binds POSITIONAL filtergraph arguments (`gblur=5`,
    `crop=100:50:10:20`) by walking the option list, skipping an entry only
    when its offset equals the one just consumed -- i.e. only when the
    duplicate is ADJACENT. The surviving list is therefore exactly ffmpeg's
    positional binding order, which is what positional call syntax binds
    against.

    Grouping by doc text alone merges genuinely distinct options that happen
    to share a description, dropping real options and shifting every
    positional slot after them. Measured: 7 of 477 filters/sources affected
    -- `deshake` (`x`/`rx`, `y`/`ry`: `x`,`y` vanished, slot 1 became `w`
    where ffmpeg binds `x`), `noise` (`all_*`/`c0_*`: six `c0_` options
    vanished, slot 4 became `c1_seed` where ffmpeg binds `c0_seed`),
    `cropdetect` (`reset`/`reset_count`, non-adjacent -- ffmpeg binds BOTH,
    slots 3 and 5), `trim`/`atrim` (`end`/`endi`/`end_pts`: `end` vanished,
    every slot from 2 on shifted), `buffer` (empty-doc options collapsed)
    and `abuffer` (`channels` vanished). Every other filter's option list is
    byte-for-byte unchanged by the adjacency rule.
  - Enum options are `<int>`-typed AVOptions whose default is often a
    CONSTANT NAME, not a number (xfade's `transition` defaults to "fade").
    Constants are the 5-space-indented lines immediately below the option,
    each `<name> <value> <flags> <description>` (description may be empty,
    e.g. overlay's `format`). `default` is stored verbatim for docs only --
    never validated against the constant list, never re-typed.
  - `(from A to B)` bounds are not always numeric (`DBL_MAX`, `INT_MAX`,
    `INT_MIN`; hqdn3d's `luma_spatial` is `(from 0 to DBL_MAX)`);
    `minimum`/`maximum` are None when a bound does not parse as a float.
  - No real `binary` or `dictionary` typed option was observed across all
    ~460 included filters. The exclusion rule for them (`type="str"`,
    `unusable=True`) is implemented but exercised only by a constructed
    fixture.
  - `boolean` options can have a non-boolean-looking default: subtitles'
    `wrap_unicode` is `<boolean> ... (default auto)`. Again, verbatim.
  - Types beyond `_TYPE_MAP` (`channel_layout`, `pix_fmt`) map to
    `"str"` via the permissive fallback -- anything not in the explicit type
    map degrades to `"str"`.
  - The flag column's FIRST character (`T` vs `.`) is retained as
    `DynamicFilter.timeline`, the FILTER's own timeline ('enable') support.
    It is independent of a per-OPTION trailing `T` in the `-help` flags
    string: `color`'s `color`/`c` option prints `..FV.....T.` although
    `color` is NOT a timeline filter (its `-filters` line is `..C` and its
    help has no "This filter has support for timeline..." sentence). Only
    the `-filters` column is read; per-option flag strings are discarded.
  - 43 source lines (`|->`) and 4 sink lines (`->|`); NONE carry `T`.
  - Source pad shapes: 40 single-output sources -- 29 `|->V` (testsrc,
    testsrc2, color, nullsrc, allrgb, mandelbrot, ...) and 11 `|->A`
    (anullsrc, sine, aevalsrc, sinc, flite, ...). One multi-output source,
    `avsynctest` (`|->AV`), and two dynamic-count, `movie`/`amovie`
    (`|->N`), are excluded exactly as multi-output/dynamic regular filters
    are: `|->AV`'s 2-char output fails the single-char check, `|->N`'s
    output char is not a V/A pad letter. All 4 sinks are single-input
    (`A->|`, `V->|`) and excluded unconditionally -- no SinkFilter exists.
  - Sources parse through the same `_parse_filter_help` (same lazy,
    memoized `Registry.options()` path) as regular filters. Verified on
    `testsrc`, `testsrc2`, `anullsrc`, `sine`, `color`: all list alias
    pairs long-name-first (size/s, rate/r, duration/d, decimals/n,
    channel_layout/cl, sample_rate/r, nb_samples/n, frequency/f,
    beep_factor/b), so the dedup rule applies unchanged.
  - The header mismatch also affects sources: `nullsrc` ->
    `"nullsrc/yuvtestsrc AVOptions:"`, `allrgb` -> `"allyuv/allrgb
    AVOptions:"` -- note neither names the queried filter first.
  - Sources print `Inputs:\n        none (source filter)` rather than
    `#0: ...`, but this module parses only the block after "AVOptions:".
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ffrwd import binaries
from ffrwd.ir import StreamType

_TIMEOUT_SECONDS = 10.0

FilterOptionType = Literal["num", "str", "bool", "duration"]


@dataclass(frozen=True)
class FilterOption:
    name: str
    type: FilterOptionType
    doc: str
    minimum: float | None
    maximum: float | None
    default: str | None  # verbatim ffmpeg text, doc use only -- never validated
    constants: tuple[str, ...]  # enum constant names, () if not an enum
    unusable: bool = False  # binary/dictionary AVOption types; lower rejects use


@dataclass(frozen=True)
class DynamicFilter:
    name: str
    inputs: tuple[StreamType, ...]  # from the pad spec, e.g. ("video", "video")
    # empty when n_input is True: the pad spec's input side is "N" (a dynamic
    # count), not a fixed list of letters.
    output: StreamType
    doc: str
    timeline: bool  # `-filters` flag column's leading `T`/`.` char
    n_input: bool = False  # input side is "N" (e.g. "N->V"): a variable number
    # of pads, all carrying `output`'s stream type -- every N->V/N->A filter
    # observed has input pads of the same kind as its single output pad.
    # Options are NOT stored here: Registry.options() loads and caches them
    # lazily per filter on first reference, never all ~460 upfront.


@dataclass(frozen=True)
class SourceFilter:
    """A zero-input (`|->V` / `|->A`) filter -- the `ffmpeg.<source>()` call.

    Multi-output (`|->AV`), dynamic-count (`|->N`) and all sinks (`->|`) are
    excluded by the pad scope check and never produce a SourceFilter. Options
    load lazily via the same `Registry.options()` path as regular filters.
    """

    name: str
    output: StreamType
    doc: str


# Guardrail #6: argv lists, a timeout, and never raise.
def _run(argv: list[str]) -> str | None:
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _get_version_line(ffmpeg: str) -> str | None:
    out = _run([ffmpeg, "-version"])
    if out is None:
        return None
    lines = out.splitlines()
    if not lines:
        return None
    return lines[0]


# A `-filters` line: one space, the 2-or-3-char flag column, the name, the pad
# spec, the description. Deliberately does NOT match the two-space-indented
# legend lines or the "Filters:" banner, so no header-skipping logic is needed.
_FILTER_LINE_RE = re.compile(r"^ ([TSC.]+) (\S+)\s+(\S+)\s+(.*)$")

_PAD_CHARS: dict[str, StreamType] = {"V": "video", "A": "audio"}


def _parse_filters_list(
    ffmpeg: str,
) -> tuple[dict[str, DynamicFilter], dict[str, SourceFilter]] | None:
    out = _run([ffmpeg, "-hide_banner", "-filters"])
    if out is None:
        return None
    filters: dict[str, DynamicFilter] = {}
    sources: dict[str, SourceFilter] = {}
    for line in out.splitlines():
        m = _FILTER_LINE_RE.match(line)
        if not m:
            continue
        flags, name, spec, doc = m.groups()
        if "->" not in spec:
            continue
        inp, _, outp = spec.partition("->")
        timeline = flags[0] == "T"
        doc_text = doc.strip()
        if inp == "|":
            # Source filter: single V/A output pad only. Multi-output
            # (`|->AV`) and dynamic-count (`|->N`) stay excluded.
            if len(outp) != 1:
                continue
            stream = _PAD_CHARS.get(outp)
            if stream is None:
                continue
            sources[name] = SourceFilter(name=name, output=stream, doc=doc_text)
            continue
        # Scope check: exclude sink (output '|'), multi-output, and
        # (defensively) zero-input specs. A dynamic pad count on the OUTPUT
        # side (`V->N`: split, `N->N`: concat/streamselect) stays excluded --
        # the compiler inserts splits itself, and a variable OUTPUT count is
        # not knowable from an option the way a variable input count is. A
        # dynamic count on the INPUT side alone (`N->V`, `N->A`) is included
        # below as n-input.
        if not inp or "|" in spec or len(outp) != 1:
            continue
        if inp == "N":
            output = _PAD_CHARS.get(outp)
            if output is None:
                continue  # N->N: dynamic on the output side too, stays excluded
            filters[name] = DynamicFilter(
                name=name,
                inputs=(),
                output=output,
                doc=doc_text,
                timeline=timeline,
                n_input=True,
            )
            continue
        try:
            inputs = tuple(_PAD_CHARS[c] for c in inp)
            output = _PAD_CHARS[outp]
        except KeyError:
            continue
        filters[name] = DynamicFilter(
            name=name, inputs=inputs, output=output, doc=doc_text, timeline=timeline
        )
    return filters, sources


_OPTION_LINE_RE = re.compile(r"^   (\S+)\s+<(\w+)>\s+(\S+)\s+(.*)$")
_CONST_LINE_RE = re.compile(r"^     (\S+)\s+(\S+)\s+(\S+)\s*(.*)$")
_RANGE_RE = re.compile(r"\(from (\S+) to (\S+)\)")
_DEFAULT_RE = re.compile(r"\(default (.*)\)\s*$")

_TYPE_MAP: dict[str, FilterOptionType] = {
    "int": "num",
    "int64": "num",
    "float": "num",
    "double": "num",
    "rational": "num",
    "boolean": "bool",
    "string": "str",
    "color": "str",
    "duration": "duration",
    "image_size": "str",
    "video_rate": "str",
    "flags": "str",
}
_UNUSABLE_TYPES = frozenset({"binary", "dictionary"})


@dataclass
class _RawOption:
    name: str
    type: str
    doc: str
    minimum: float | None
    maximum: float | None
    default: str | None
    constants: list[str] = field(default_factory=list)


def _try_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _parse_option_block(lines: list[str]) -> dict[str, FilterOption]:
    """Parse option/constant lines up to the first blank line (or EOF)."""
    raw: list[_RawOption] = []
    current: _RawOption | None = None
    for line in lines:
        if line.strip() == "":
            break
        indent = len(line) - len(line.lstrip(" "))
        if indent >= 5 and current is not None:
            m = _CONST_LINE_RE.match(line)
            if m:
                current.constants.append(m.group(1))
            continue
        if indent == 3:
            m2 = _OPTION_LINE_RE.match(line)
            if m2:
                oname, otype, _flags, rest = m2.groups()
                default = None
                dm = _DEFAULT_RE.search(rest)
                if dm:
                    default = dm.group(1).strip()
                minimum = maximum = None
                rm = _RANGE_RE.search(rest)
                if rm:
                    minimum = _try_float(rm.group(1))
                    maximum = _try_float(rm.group(2))
                doc = _DEFAULT_RE.sub("", _RANGE_RE.sub("", rest)).strip()
                current = _RawOption(oname, otype, doc, minimum, maximum, default)
            else:
                # Option-shaped line that isn't `<name> <type> <flags> desc`:
                # degrade to a bare str option rather than drop the filter.
                parts = line.strip().split(None, 1)
                if not parts:
                    current = None
                    continue
                oname = parts[0]
                doc = parts[1] if len(parts) > 1 else ""
                current = _RawOption(oname, "string", doc, None, None, None)
            raw.append(current)
            continue
        # Any other indentation: ignore it without disturbing `current`.
    return _dedup_and_convert(raw)


def _dedup_and_convert(raw: list[_RawOption]) -> dict[str, FilterOption]:
    # Collapse each run of CONSECUTIVE same-doc lines (ffmpeg's short/long
    # alias signal) to the longest name in the run, ties by first occurrence.
    # Adjacency is the whole rule: ffmpeg's positional binding skips a
    # duplicate only when it immediately follows the entry it aliases, so
    # NON-adjacent same-doc options are two real options and both keep their
    # slot. See the module docstring for the seven filters this decides.
    keep: list[int] = []
    start = 0
    while start < len(raw):
        end = start
        while end + 1 < len(raw) and raw[end + 1].doc == raw[start].doc:
            end += 1
        best = start
        for i in range(start + 1, end + 1):
            if len(raw[i].name) > len(raw[best].name):
                best = i
        keep.append(best)
        start = end + 1

    result: dict[str, FilterOption] = {}
    for i in keep:
        o = raw[i]
        unusable = o.type in _UNUSABLE_TYPES
        if unusable or o.constants:
            ftype: FilterOptionType = "str"
        else:
            ftype = _TYPE_MAP.get(o.type, "str")
        result[o.name] = FilterOption(
            name=o.name,
            type=ftype,
            doc=o.doc,
            minimum=o.minimum,
            maximum=o.maximum,
            default=o.default,
            constants=tuple(o.constants),
            unusable=unusable,
        )
    return result


def _parse_filter_help(ffmpeg: str, name: str) -> dict[str, FilterOption]:
    out = _run([ffmpeg, "-hide_banner", "-help", f"filter={name}"])
    if out is None:
        return {}
    lines = out.splitlines()
    start = None
    for i, line in enumerate(lines):
        if "AVOptions:" in line:
            start = i + 1
            break
    if start is None:
        return {}
    return _parse_option_block(lines[start:])


# Disk cache: ~/.cache/ffrwd/, keyed by a hash of `ffmpeg -version`.
#
# Bump on any change to the cached payload shape. A mismatch -- including an
# absent key, as in older cache files -- is treated exactly like corrupt JSON:
# silently discarded and rebuilt from a fresh `-filters`/`-help` pass.
#
# 3: `DynamicFilter` gained `n_input` (a dynamic INPUT pad count, `N->V`/
# `N->A`, is now included rather than excluded -- see `_parse_filters_list`).
_CACHE_FORMAT_VERSION = 3


def _cache_dir() -> Path:
    try:
        return Path.home() / ".cache" / "ffrwd"
    except RuntimeError:  # pragma: no cover -- no resolvable home directory
        return Path(tempfile.gettempdir()) / "ffrwd-cache"


def _cache_path(version_line: str) -> Path:
    digest = hashlib.sha256(version_line.encode("utf-8")).hexdigest()[:16]
    return _cache_dir() / f"registry-{digest}.json"


def _require_str(v: object) -> str:
    if not isinstance(v, str):
        raise ValueError("expected str")
    return v


def _require_optional_str(v: object) -> str | None:
    if v is None:
        return None
    return _require_str(v)


def _require_bool(v: object) -> bool:
    if not isinstance(v, bool):
        raise ValueError("expected bool")
    return v


def _require_optional_float(v: object) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError("expected float")
    return float(v)


def _require_list(v: object) -> list[object]:
    if not isinstance(v, list):
        raise ValueError("expected list")
    return v


def _require_option_type(v: object) -> FilterOptionType:
    if v == "num":
        return "num"
    if v == "str":
        return "str"
    if v == "bool":
        return "bool"
    if v == "duration":
        return "duration"
    raise ValueError(f"bad option type {v!r}")


def _require_stream_type(v: object) -> StreamType:
    if v == "video":
        return "video"
    if v == "audio":
        return "audio"
    raise ValueError(f"bad stream type {v!r}")


@dataclass
class _DiskCache:
    filters: dict[str, DynamicFilter]
    sources: dict[str, SourceFilter]
    options: dict[str, dict[str, FilterOption]]


def _decode_filters(raw: object) -> dict[str, DynamicFilter]:
    if not isinstance(raw, dict):
        raise ValueError("filters not a dict")
    result: dict[str, DynamicFilter] = {}
    for name, entry in raw.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise ValueError("bad filter entry")
        inputs_raw = _require_list(entry["inputs"])
        inputs = tuple(_require_stream_type(x) for x in inputs_raw)
        output = _require_stream_type(entry["output"])
        doc = _require_str(entry["doc"])
        timeline = _require_bool(entry["timeline"])
        n_input = _require_bool(entry["n_input"])
        result[name] = DynamicFilter(
            name=name, inputs=inputs, output=output, doc=doc, timeline=timeline, n_input=n_input
        )
    return result


def _decode_sources(raw: object) -> dict[str, SourceFilter]:
    if not isinstance(raw, dict):
        raise ValueError("sources not a dict")
    result: dict[str, SourceFilter] = {}
    for name, entry in raw.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise ValueError("bad source entry")
        output = _require_stream_type(entry["output"])
        doc = _require_str(entry["doc"])
        result[name] = SourceFilter(name=name, output=output, doc=doc)
    return result


def _decode_option(raw: object) -> FilterOption:
    if not isinstance(raw, dict):
        raise ValueError("option not a dict")
    return FilterOption(
        name=_require_str(raw["name"]),
        type=_require_option_type(raw["type"]),
        doc=_require_str(raw["doc"]),
        minimum=_require_optional_float(raw["minimum"]),
        maximum=_require_optional_float(raw["maximum"]),
        default=_require_optional_str(raw["default"]),
        constants=tuple(_require_str(c) for c in _require_list(raw["constants"])),
        unusable=_require_bool(raw["unusable"]),
    )


def _decode_options(raw: object) -> dict[str, dict[str, FilterOption]]:
    if not isinstance(raw, dict):
        raise ValueError("options not a dict")
    result: dict[str, dict[str, FilterOption]] = {}
    for filter_name, opts in raw.items():
        if not isinstance(filter_name, str) or not isinstance(opts, dict):
            raise ValueError("bad options entry")
        result[filter_name] = {
            opt_name: _decode_option(opt)
            for opt_name, opt in opts.items()
            if isinstance(opt_name, str)
        }
    return result


def _decode_payload(data: dict[str, object]) -> _DiskCache:
    """Shared shape decode for both the disk cache AND the reference snapshot.

    Raises (KeyError, TypeError, ValueError) on a malformed shape; both
    callers (`_read_disk_cache`, `load_reference`) catch those and degrade
    permissively, per this module's NEVER-raises contract. Ignores
    `format_version`/`version_line` -- the freshness rules differ: the disk
    cache pins BOTH to the live binary, the reference snapshot checks only
    `format_version` since it deliberately outlives any one binary.
    """
    filters = _decode_filters(data["filters"])
    sources = _decode_sources(data["sources"])
    options = _decode_options(data["options"])
    return _DiskCache(filters=filters, sources=sources, options=options)


def _encode_payload(
    version_line: str,
    filters: dict[str, DynamicFilter],
    sources: dict[str, SourceFilter],
    options: dict[str, dict[str, FilterOption]],
) -> dict[str, object]:
    """The disk-cache-shaped payload dict, JSON-ready, shared by the disk
    cache writer and `Registry.to_snapshot_payload()` (`scripts/gen_snapshot.py`)."""
    return {
        "format_version": _CACHE_FORMAT_VERSION,
        "version_line": version_line,
        "filters": {
            name: {
                "inputs": list(f.inputs),
                "output": f.output,
                "doc": f.doc,
                "timeline": f.timeline,
                "n_input": f.n_input,
            }
            for name, f in filters.items()
        },
        "sources": {
            name: {"output": s.output, "doc": s.doc} for name, s in sources.items()
        },
        "options": {
            filter_name: {
                opt_name: {
                    "name": o.name,
                    "type": o.type,
                    "doc": o.doc,
                    "minimum": o.minimum,
                    "maximum": o.maximum,
                    "default": o.default,
                    "constants": list(o.constants),
                    "unusable": o.unusable,
                }
                for opt_name, o in opts.items()
            }
            for filter_name, opts in options.items()
        },
    }


def _read_disk_cache(version_line: str) -> _DiskCache | None:
    path = _cache_path(version_line)
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw_text)
        if not isinstance(data, dict):
            return None
        if data.get("format_version") != _CACHE_FORMAT_VERSION:
            return None  # absent or mismatched: rebuild, never guess the shape
        if data.get("version_line") != version_line:
            return None
        return _decode_payload(data)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_disk_cache(
    version_line: str,
    filters: dict[str, DynamicFilter],
    sources: dict[str, SourceFilter],
    options: dict[str, dict[str, FilterOption]],
) -> None:
    data = _encode_payload(version_line, filters, sources, options)
    # Purely an optimization: any filesystem failure is swallowed.
    try:
        cache_dir = _cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=cache_dir, prefix="registry-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            os.replace(tmp_name, _cache_path(version_line))
        except OSError:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
    except OSError:
        pass


class Registry:
    """Lazily-loaded view of the installed ffmpeg's filter set.

    `-filters` is parsed at most once per process, on first call to any of
    `available()`, `get()`, `names()`, `get_source()`, `source_names()`,
    `options()` or `excluded_options()`. Per-filter `-help filter=X` is parsed
    at most once per filter (regular OR source), on first call to
    `options(name)` / `excluded_options(name)` for that filter -- never for all
    filters upfront. NEVER raises.

    `source` marks where the data came from: `"live"` (default, via `load()`
    or a bare `Registry()` -- lazy introspection of the ffmpeg on PATH) or
    `"reference"` (via `load_reference()` -- fully populated up front from the
    vendored snapshot, no subprocess ever). `snapshot_of` (the snapshot
    ffmpeg's `-version` first line) and `generated` (the `--stamp` value
    `scripts/gen_snapshot.py` ran with) are set only on a `"reference"`
    instance, None on `"live"`. Callers read `source` to choose
    live-vs-snapshot and to annotate `explain` output.
    """

    def __init__(self) -> None:
        self._ffmpeg: str | None = None
        self._version_line: str | None = None
        self._loaded = False
        self._filters: dict[str, DynamicFilter] = {}
        self._sources: dict[str, SourceFilter] = {}
        self._options: dict[str, dict[str, FilterOption]] = {}
        self.source: Literal["live", "reference"] = "live"
        self.snapshot_of: str | None = None
        self.generated: str | None = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        ffmpeg = binaries.ffmpeg_path()
        if ffmpeg is None:
            return
        version_line = _get_version_line(ffmpeg)
        if version_line is None:
            return
        self._ffmpeg = ffmpeg
        self._version_line = version_line

        disk = _read_disk_cache(version_line)
        if disk is not None:
            self._filters = disk.filters
            self._sources = disk.sources
            self._options = disk.options
            return

        parsed = _parse_filters_list(ffmpeg)
        if parsed is None:
            return
        filters, sources = parsed
        if not filters and not sources:
            return
        self._filters = filters
        self._sources = sources
        _write_disk_cache(version_line, self._filters, self._sources, self._options)

    def available(self) -> bool:
        """True if ffmpeg was found on PATH and `-filters` parsed."""
        self._ensure_loaded()
        return bool(self._filters) or bool(self._sources)

    def names(self) -> list[str]:
        """All included (non-excluded) regular filter names, ffmpeg's own order.

        Sources are NOT included (use `source_names()`): this list drives
        column-function lookup, and a source is not callable as a column
        function.
        """
        self._ensure_loaded()
        return list(self._filters)

    def get(self, name: str) -> DynamicFilter | None:
        """None if `name` is unknown to this ffmpeg OR excluded by the pad scope check.

        Includes n-input filters (`DynamicFilter.n_input`, a dynamic INPUT pad
        count) -- only a dynamic OUTPUT count, multi-output, sink or source
        spec is excluded. Also None for a known SOURCE name: a source is not a
        column function, use `get_source()`.
        """
        self._ensure_loaded()
        return self._filters.get(name)

    def get_source(self, name: str) -> SourceFilter | None:
        """None if `name` is unknown, a sink, a multi-output, or a dynamic-pad source."""
        self._ensure_loaded()
        return self._sources.get(name)

    def source_names(self) -> list[str]:
        """All included (single V/A output) source names, ffmpeg's own order."""
        self._ensure_loaded()
        return list(self._sources)

    def options(self, name: str) -> dict[str, FilterOption] | None:
        """None if `name` is unknown; {} if known but has no (parseable) options.

        Works for both regular filters and sources (same lazy, memoized
        `-help filter=X` path either way).
        """
        self._ensure_loaded()
        if name not in self._filters and name not in self._sources:
            return None
        cached = self._options.get(name)
        if cached is not None:
            return cached
        if self._ffmpeg is None:
            return {}
        opts = _parse_filter_help(self._ffmpeg, name)
        self._options[name] = opts
        if self._version_line is not None:
            _write_disk_cache(self._version_line, self._filters, self._sources, self._options)
        return opts

    def excluded_options(self, name: str) -> dict[str, FilterOption] | None:
        """`-help filter=<name>` options for a name the v1 pad scope check EXCLUDED.

        `options()` answers only for names that SURVIVED that check, i.e. the
        keys of `_filters`/`_sources`; an excluded name reads as unknown
        there. The array-RETURNING filters (`channelsplit`, `acrossover`,
        `extractplanes` -- all `->N`) are excluded from those tables yet
        callable through table lowering, and this is their one door:
        the same lazy, memoized, permissive `-help` path, implying no pad
        information.

        None means "this ffmpeg cannot tell me about that filter": no ffmpeg,
        or `-help filter=<name>` printed no option block -- which is what a
        build WITHOUT the filter prints (`Unknown filter 'x'.`, exit 0, no
        "AVOptions:" line; verified against ffmpeg 7.1). Every name this
        accessor exists for has a non-empty option table in a build that has
        it, so lowering reads None as "not in this build" and rejects the call
        as an unknown function. That inference is this accessor's contract,
        not a general one -- it does not hold for a filter with no options.
        """
        self._ensure_loaded()
        cached = self._options.get(name)
        if cached is not None:
            return cached
        if self._ffmpeg is None:
            return None
        opts = _parse_filter_help(self._ffmpeg, name)
        if not opts:
            return None
        self._options[name] = opts
        if self._version_line is not None:
            _write_disk_cache(self._version_line, self._filters, self._sources, self._options)
        return opts

    def to_snapshot_payload(self) -> dict[str, object] | None:
        """This Registry's CURRENT state as a disk-cache-shaped JSON payload.

        `None` if `-filters`/`-version` never succeeded (no version line to
        stamp with). Forces nothing to load first, so a payload taken from a
        `Registry` nobody has queried has an empty `options` table.
        `scripts/gen_snapshot.py`, the only intended caller, force-loads every
        name's options first, which is what makes its payload self-contained.
        """
        self._ensure_loaded()
        if self._version_line is None:
            return None
        return _encode_payload(self._version_line, self._filters, self._sources, self._options)


_registry: Registry | None = None


def load() -> Registry:
    """Return the process-wide Registry singleton (memoized)."""
    global _registry
    if _registry is None:
        _registry = Registry()
    return _registry


def load_reference(path: str | Path) -> Registry:
    """Build a fully-populated `Registry` from a reference snapshot FILE.

    `path` is explicit and required: the snapshot is a committed TEST FIXTURE
    (`tests/data/reference_registry.json`), not package data, and nothing in
    the installed package reads it. No implicit lookup, no
    `importlib.resources`, so an installed ffrwd does not carry the file.

    The payload is the parsed `-filters`/`-help` data
    `scripts/gen_snapshot.py` captured from a real ffmpeg, wrapped with
    `snapshot_of` (that ffmpeg's `-version` first line) and `generated` (the
    `--stamp` it ran with). Every in-scope filter's and source's options, plus
    the array-returning trio's `excluded_options()`, are already in it -- so
    `_ensure_loaded()` never runs (`_loaded` is True up front) and the
    returned instance spawns NO subprocess, on any platform, with or without
    ffmpeg on PATH.

    Returns a FRESH `Registry` every call, unlike `load()`'s singleton.

    NEVER raises: a missing or malformed snapshot degrades to an empty,
    unavailable `Registry` with `source == "reference"`, exactly like a live
    `Registry` with no ffmpeg on PATH.
    """
    registry = Registry()
    registry.source = "reference"
    registry._loaded = True  # never let _ensure_loaded touch which()/subprocess
    try:
        raw_text = Path(path).read_text(encoding="utf-8")
        data = json.loads(raw_text)
        if not isinstance(data, dict):
            return registry
        if data.get("format_version") != _CACHE_FORMAT_VERSION:
            return registry
        disk = _decode_payload(data)
        version_line = _require_optional_str(data.get("version_line"))
        snapshot_of = _require_optional_str(data.get("snapshot_of"))
        generated = _require_optional_str(data.get("generated"))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return registry
    registry._filters = disk.filters
    registry._sources = disk.sources
    registry._options = disk.options
    registry._version_line = version_line
    registry.snapshot_of = snapshot_of
    registry.generated = generated
    return registry


def clear_cache() -> None:
    """Reset the process memo AND remove on-disk registry cache file(s). For tests."""
    global _registry
    _registry = None
    try:
        cache_dir = _cache_dir()
        if cache_dir.is_dir():
            for entry in cache_dir.iterdir():
                if entry.name.startswith("registry-") and entry.name.endswith(".json"):
                    try:
                        entry.unlink()
                    except OSError:
                        pass
    except OSError:
        pass
