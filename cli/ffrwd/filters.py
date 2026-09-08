"""Finding a filter in the registry, and what to say when it is not there.

The local ffmpeg is introspected once (:mod:`ffrwd.registry`); this is what
asks it for one filter, for the option table a call needs, and for the
near-miss names a refusal offers when the written name is not among them.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from dataclasses import dataclass

from sqlglot import exp

from ffrwd import binaries
from ffrwd.calls import _Call
from ffrwd.errors import ErrorCode
from ffrwd.expressions import _error
from ffrwd.filter_options import _listed
from ffrwd.ir import StreamType
from ffrwd.parser import FILTER_NAMESPACE, RawSource
from ffrwd.registry import DynamicFilter, FilterOption, Registry, SourceFilter


def _filter_options(
    registry: Registry | None, filter_name: str, anchor: exp.Expr, fallback: exp.Expr
) -> dict[str, FilterOption]:
    """The introspected options of `filter_name`, or a typed rejection.

    One rule: options ARE the installed ffmpeg. Without a registry there is
    nothing to validate them against, and guessing is exactly what this
    compiler does not do. (A CALL cannot reach this with a None registry —
    its name would already be UNKNOWN_FUNCTION — but a generated source in
    FROM position can, so the branch stays.)

    ``Registry.options`` returns None only for a filter this ffmpeg does not
    have (or that the v1 scope check excluded); an empty dict is a real
    answer (a filter with no options) and is passed through as one.
    """
    if registry is None:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            "options are validated against your installed ffmpeg; "
            "the provisioner failed to supply one",
            anchor,
            fallback=fallback,
            hint=_NO_REGISTRY_HINT,
        )
    options = registry.options(filter_name)
    if options is None:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"options are validated against the ffmpeg filter "
            f"'{filter_name}', which your ffmpeg does not provide",
            anchor,
            fallback=fallback,
            hint="drop the options, or install an ffmpeg that has "
            f"the '{filter_name}' filter",
        )
    return options


def _options_for(
    registry: Registry | None,
    filter_name: str,
    call: _Call,
    stream_arity: int,
    node: exp.Expr,
    select: exp.Select,
) -> dict[str, FilterOption]:
    """The filter's option table, fetched only when the call actually needs it.

    ``-help filter=X`` is a subprocess, and a call that passes no options
    at all (``hflip(a.video[1])``) has nothing to validate — so the table stays
    unfetched, exactly as it did before positional options existed.
    """
    if len(call.args) <= stream_arity and not call.named:
        return {}
    return _filter_options(registry, filter_name, node, select)


def _unknown_function_hint(registry: Registry | None, name: str) -> str:
    """Did-you-mean over the registry (there is nothing else)."""
    if registry is not None and registry.available():
        if name == _CONCAT_NAME:
            return _CONCAT_VARIADIC_HINT
        if registry.get_source(name) is not None:
            return (
                f"{name} is a generated source, not a function: put it in FROM, "
                f"e.g. FROM {FILTER_NAMESPACE}.{name}(duration => 2) s"
            )
        # An n-input filter (amix, hstack, xstack, ...) is already in
        # `registry.names()` -- an ordinary registry member now -- so
        # only `concat` (excluded on the OUTPUT side) needs adding by hand.
        candidates = sorted((set(registry.names()) | {_CONCAT_NAME}) - {name})
        matches = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
        if matches:
            return f"did you mean {matches[0]}()?"
        return (
            "every function is a filter of your installed ffmpeg, and this is "
            "not one of them; filters with a variable pad count, more than one "
            "output, or no input at all are not callable"
        )
    return _NO_REGISTRY_HINT


def _namespaced_function_hint(registry: Registry | None, name: str) -> str:
    """Did-you-mean for ``ffmpeg.<filter>()``, keeping the namespace spelling.

    Suggestions keep the ``ffmpeg.`` prefix, which is the one spelling that
    works for every filter name whatever Postgres thinks of it.
    """
    if registry is not None and registry.available():
        if name == _CONCAT_NAME:
            return _CONCAT_VARIADIC_HINT
        if registry.get_source(name) is not None:
            # A generated source IS usable -- in FROM, where it belongs
            #. Say where rather than "unknown".
            return (
                f"{FILTER_NAMESPACE}.{name} is a generated source, not a "
                f"function: put it in FROM, e.g. FROM {FILTER_NAMESPACE}."
                f"{name}(duration => 2) s"
            )
        candidates = sorted(
            (set(registry.names()) | set(ARRAY_RETURNING) | {_CONCAT_NAME}) - {name}
        )
        matches = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
        if matches:
            return f"did you mean {FILTER_NAMESPACE}.{matches[0]}()?"
        return (
            f"{FILTER_NAMESPACE}.<filter> is a filter of your installed ffmpeg, "
            "and this is not one of them; filters with a variable pad count, "
            "more than one output, or no input at all are not callable"
        )
    return (
        f"the {FILTER_NAMESPACE}.<filter> namespace is your installed ffmpeg's "
        "filter set; the provisioner failed to supply one"
    )


def _unknown_source_hint(registry: Registry | None, name: str) -> str:
    """Did-you-mean over ``source_names()``, then why the set might be missing.

    Mirrors :meth:`_namespaced_function_hint` branch for branch — the
    namespace is the same one, and a source is unavailable for exactly the
    same reasons a namespaced call is — but suggests only SOURCES, since
    a regular filter would not be usable in FROM either way.
    """
    if registry is not None and registry.available():
        matches = difflib.get_close_matches(
            name, sorted(registry.source_names()), n=1, cutoff=0.6
        )
        if matches:
            return f"did you mean {FILTER_NAMESPACE}.{matches[0]}()?"
        return (
            f"FROM {FILTER_NAMESPACE}.<source>(...) takes a zero-input filter of "
            "your installed ffmpeg, and this is not one of them; sources with "
            "more than one output pad (avsynctest) or a variable pad count "
            "(movie, amovie) are not usable"
        )
    return (
        f"FROM {FILTER_NAMESPACE}.<source>(...) generates a stream with your "
        "installed ffmpeg; the provisioner failed to supply one"
    )


def _source_filter(registry: Registry | None, raw: RawSource, select: exp.Select) -> SourceFilter:
    """The registry's entry for ``ffmpeg.<name>`` in FROM position, or a rejection.

    Three ways this fails, in the order they are told apart:

    * the name is a REGULAR filter of this ffmpeg (``ffmpeg.gblur``) — it
      has input pads, so it is a call, not a table: UNSUPPORTED_SQL saying
      so, the one excluded case that is positively identifiable;
    * there is no registry at all (no ffmpeg) — the standard
      unavailability wording, same as a namespaced CALL's;
    * the name is unknown to both tables — UNKNOWN_FUNCTION with a
      did-you-mean over ``source_names()``. Sources the v1 scope check
      excluded (``avsynctest``'s ``|->AV``, ``movie``/``amovie``'s
      ``|->N``) are NOT retained by the registry at all, so they are
      indistinguishable from a typo here and land on the same rejection —
      which is why its fallback hint states the exclusion explicitly rather
      than only listing near-misses.
    """
    source = registry.get_source(raw.name) if registry is not None else None
    if source is not None:
        return source
    if registry is not None and registry.get(raw.name) is not None:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"{FILTER_NAMESPACE}.{raw.name} is an ffmpeg filter, not a source: "
            "it takes stream inputs, so it cannot stand in FROM",
            raw.call_node,
            fallback=select,
            hint=f"call it over a stream instead, e.g. SELECT "
            f"{FILTER_NAMESPACE}.{raw.name}(a.video[1]) FROM input('clip.mp4') a",
        )
    raise _error(
        ErrorCode.UNKNOWN_FUNCTION,
        f"unknown generated source {FILTER_NAMESPACE}.{raw.name}()",
        raw.call_node,
        fallback=select,
        hint=_unknown_source_hint(registry, raw.name),
    )


def _n_input_call(
    registry: Registry | None, name: str
) -> tuple[_NInputFilter, dict[str, FilterOption]] | None:
    """`name`'s derived call shape and option table, if THIS registry has
    it as a callable N-input filter (``DynamicFilter.n_input``).

    An n-input filter is an ordinary registry member now (see
    registry.py), so this is a membership check plus the same option
    fetch every other callable filter goes through -- options are
    fetched even for a call that passes none, since the spec's
    `option`/`fallback` are derived from the table's own content
    (:func:`_n_input_spec`). ``acrossfade`` is the case this matters for:
    on a build where it is still an ordinary ``AA->A`` filter,
    ``dynamic.n_input`` is False and this returns None, so the registry's
    own pad signature wins over any N-input treatment.
    """
    if registry is None:
        return None
    dynamic = registry.get(name)
    if dynamic is None or not dynamic.n_input:
        return None
    options = registry.options(name)
    if options is None:
        return None
    return _n_input_spec(name, dynamic, options), options


def _twin_pair(
    registry: Registry | None, name: str, dynamic: DynamicFilter
) -> DynamicFilter | None:
    """``name``'s audio twin ``a<name>``, if the pair is eligible, else None.

    Eligibility comes straight from the registry, not a curated list:
    ``name`` takes video-only input, ``a<name>`` exists and takes
    audio-only input. That excludes a pair that only shares a stem, like
    ``interleave``/``ainterleave`` or ``mix``/``amix`` (both N-input, so
    neither has a fixed pad type to compare). Shared between the twin
    dispatch itself and the refusal hint that names a stem, so the two
    never disagree about what counts as a pair.
    """
    if registry is None:
        return None
    if dynamic.n_input or not dynamic.inputs or any(k != "video" for k in dynamic.inputs):
        return None
    twin = registry.get("a" + name)
    if twin is None or twin.n_input or not twin.inputs:
        return None
    if any(k != "audio" for k in twin.inputs):
        return None
    return twin


def _twin_dispatch_stem(
    registry: Registry | None, name: str, call: _Call, got: list[str]
) -> str | None:
    """The video stem to name in a refusal, when a hand-spelled audio
    twin was handed a video stream, or None for the ordinary hint.

    Only ``a<stem>(video)`` qualifies: a bare call (not
    ``ffmpeg.a<stem>(...)``, which is exact and never switches), whose
    name starts with ``a``, whose stem is a video-only filter with
    exactly this name as its eligible audio twin (:meth:`_twin_pair`),
    and whose first stream argument is video -- the shape the twin
    dispatch would have picked up had the query spelled the bare stem
    instead.
    """
    if call.namespaced or registry is None:
        return None
    if len(name) < 2 or not name.startswith("a") or got[:1] != ["video"]:
        return None
    stem = name[1:]
    video = registry.get(stem)
    if video is None:
        return None
    return stem if _twin_pair(registry, stem, video) is not None else None


def _array_options(registry: Registry | None, name: str) -> dict[str, FilterOption] | None:
    """`name`'s option table if it is a callable array-returning filter.

    Three questions, one answer, because they have the same shape: is the
    name in :data:`ARRAY_RETURNING`, is there a registry at all, and does
    THIS ffmpeg actually have the filter. The last one is why the
    options are fetched even for a call with no named arguments: an excluded
    name is in no registry table, so its option block is the only evidence
    this build has it (see ``Registry.excluded_options``). None means "not
    callable", and the caller falls through to the ordinary namespaced
    rejection, hint and all.
    """
    if name not in ARRAY_RETURNING or registry is None:
        return None
    return registry.excluded_options(name)


def _concat_options(registry: Registry | None, name: str) -> dict[str, FilterOption] | None:
    """``concat``'s option table, but ONLY for a call under VARIADIC.

    Mirrors :meth:`_n_input_options`: ``concat`` is ``N->N`` and excluded
    from the registry's own table by the pad-scope check (see
    registry.py), and ``excluded_options`` is the one door back in --
    also the evidence that this ffmpeg actually ships the filter at all.
    """
    if name != _CONCAT_NAME or registry is None:
        return None
    return registry.excluded_options(name)


_NO_REGISTRY_HINT = (
    f"ffrwd's function surface IS your installed ffmpeg's filter set; {binaries.INSTALL_HINT}"
)


@dataclass(frozen=True)
class _BadCount:
    """A count rule's rejection: which option said what, and what was expected."""

    option: str
    value: str
    expected: str
    hint: str


@dataclass(frozen=True)
class _ArrayFilter:
    """One array-returning filter: its pads, and how an option fixes its count."""

    name: str
    input: StreamType  # its single input pad
    element: StreamType  # what every one of its output pads carries
    count: Callable[[dict[str, object]], int | _BadCount]


# `ffmpeg -layouts` (7.1), "Standard channel layouts": name -> how many
# channels its decomposition lists. Data, verbatim -- the whole table ffmpeg
# printed, not a curated subset of it, so the only layouts a query can be
# rejected for are the ones this ffmpeg would reject too.
_CHANNEL_LAYOUTS: dict[str, int] = {
    "mono": 1,
    "stereo": 2,
    "2.1": 3,
    "3.0": 3,
    "3.0(back)": 3,
    "4.0": 4,
    "quad": 4,
    "quad(side)": 4,
    "3.1": 4,
    "5.0": 5,
    "5.0(side)": 5,
    "4.1": 5,
    "5.1": 6,
    "5.1(side)": 6,
    "6.0": 6,
    "6.0(front)": 6,
    "3.1.2": 6,
    "hexagonal": 6,
    "6.1": 7,
    "6.1(back)": 7,
    "6.1(front)": 7,
    "7.0": 7,
    "7.0(front)": 7,
    "7.1": 8,
    "7.1(wide)": 8,
    "7.1(wide-side)": 8,
    "5.1.2": 8,
    "octagonal": 8,
    "cube": 8,
    "5.1.4": 10,
    "7.1.2": 10,
    "7.1.4": 12,
    "7.2.3": 12,
    "9.1.4": 14,
    "hexadecagonal": 16,
    "downmix": 2,
    "22.2": 24,
}


# `ffmpeg -layouts` (7.1), "Individual channels": the names a custom layout is
# composed of with `+` (`FL+FR`, `FC+LFE`), which ffmpeg accepts anywhere a
# standard layout name is accepted.
_CHANNEL_NAMES: frozenset[str] = frozenset(
    {
        "FL", "FR", "FC", "LFE", "BL", "BR", "FLC", "FRC", "BC", "SL", "SR",
        "TC", "TFL", "TFC", "TFR", "TBL", "TBC", "TBR", "DL", "DR", "WL", "WR",
        "SDL", "SDR", "LFE2", "TSL", "TSR", "BFC", "BFL", "BFR", "SSL", "SSR",
        "TTL", "TTR",
    }
)


_LAYOUT_HINT = (
    "a channel layout is one of ffmpeg's standard names (see `ffmpeg -layouts`) "
    "or a '+'-joined list of channel names, e.g. 'stereo', '5.1', 'FL+FR'"
)


_SPLIT_HINT = (
    "acrossover splits at a list of positive frequencies separated by spaces or "
    "'|', e.g. split => '500' (2 bands) or split => '500|3000' (3 bands)"
)


_PLANES_HINT = (
    "planes names the planes to extract, e.g. planes => 'y'; your ffmpeg types "
    "it as an enum, so only ONE plane per call is accepted here"
)


def _option_text(value: object) -> str:
    """A validated option value as the text ffmpeg will be handed."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _channel_count(text: str) -> int | None:
    """How many channels a layout spelling describes, or None if unrecognized."""
    standard = _CHANNEL_LAYOUTS.get(text)
    if standard is not None:
        return standard
    parts = text.split("+")
    if parts and all(part in _CHANNEL_NAMES for part in parts):
        return len(parts)
    return None


def _channelsplit_count(args: dict[str, object]) -> int | _BadCount:
    """One output pad per channel channelsplit is asked to extract.

    `channels` (default "all") wins when it is set to anything else: it is
    itself a layout spelling naming the SUBSET to split out, so
    `channels => 'FL'` is one pad however wide `channel_layout` is. Verified
    against ffmpeg 7.1 -- a graph that labels more pads than the filter has is
    a hard "More output link labels specified ... than it has outputs" error,
    so the count has to follow both options, not just the documented one.
    """
    channels = _option_text(args.get("channels", "all"))
    if channels != "all":
        count = _channel_count(channels)
        if count is None:
            return _BadCount("channels", channels, "a channel layout", _LAYOUT_HINT)
        return count
    layout = _option_text(args.get("channel_layout", "stereo"))
    count = _channel_count(layout)
    if count is None:
        return _BadCount(
            "channel_layout",
            layout,
            f"one of {_listed(_CHANNEL_LAYOUTS)}",
            _LAYOUT_HINT,
        )
    return count


def _acrossover_count(args: dict[str, object]) -> int | _BadCount:
    """One band per split frequency, plus the band below the lowest one."""
    split = _option_text(args.get("split", "500"))
    parts = split.replace("|", " ").split()
    ok = bool(parts)
    for part in parts:
        try:
            frequency = float(part)
        except ValueError:
            ok = False
            break
        if not frequency > 0:
            ok = False
            break
    if not ok:
        return _BadCount("split", split, "a list of positive frequencies", _SPLIT_HINT)
    return len(parts) + 1


def _extractplanes_count(args: dict[str, object]) -> int | _BadCount:
    """One output pad per requested plane.

    ffmpeg's own option is a `flags` set (`y+u+v`), but the registry types an
    option that lists constants as an enum, so `_option_value` accepts exactly
    one of them and a `+`-joined value is FILTER_OPTION_TYPE before this rule
    ever runs. The `+` arithmetic is written out anyway: it is what the option
    means, and it is what a later plan widening flags handling will need.
    """
    planes = _option_text(args.get("planes", "r"))
    parts = planes.split("+")
    if not parts or not all(parts):
        return _BadCount("planes", planes, "one or more plane names", _PLANES_HINT)
    return len(parts)


ARRAY_RETURNING: dict[str, _ArrayFilter] = {
    "channelsplit": _ArrayFilter(
        name="channelsplit",
        input="audio",
        element="audio",
        count=_channelsplit_count,
    ),
    "acrossover": _ArrayFilter(
        name="acrossover",
        input="audio",
        element="audio",
        count=_acrossover_count,
    ),
    "extractplanes": _ArrayFilter(
        name="extractplanes",
        input="video",
        element="video",
        count=_extractplanes_count,
    ),
}


_ARRAY_INPUT_HINT = (
    "an array-returning filter takes exactly one stream, because its own result "
    "is the array; subscript the argument, e.g. a.audio[1]"
)


@dataclass(frozen=True)
class _NInputFilter:
    """One N-input filter's call shape: its pads, and the option fixing the count."""

    name: str
    stream: StreamType  # what every one of its INPUT pads carries
    output: StreamType  # its single output pad
    option: str | None  # the option whose value IS the input-pad count; None
    # when there is no such option (ladspa: the plugin's own ports decide) --
    # then the supplied stream count is never checked against anything and
    # never written back.
    fallback: int  # count when the option is neither written nor introspectable
    # Write the count onto the node even when it equals the fallback. True for
    # the filters that are N-input on EVERY ffmpeg (amix: pins carry
    # `inputs=2`); False for ones that grew the option in a later ffmpeg
    # (acrossfade, N->A since ffmpeg 9) -- omitting the defaulted count keeps
    # the compiled command valid on builds whose acrossfade has no such
    # option at all.
    emit_default: bool = True


# What no single ffmpeg build's introspection can answer about itself:
# whether writing the DEFAULTED count is safe on an older build that lacks
# the option entirely (acrossfade, N->A only since ffmpeg 9), and ladspa's
# fallback/emit_default, whose "count" is never a real ffmpeg option value.
# Everything else is derived from the registry -- see `_n_input_spec`.
@dataclass(frozen=True)
class _NInputOverride:
    fallback: int | None = None
    emit_default: bool | None = None


_N_INPUT_OVERRIDES: dict[str, _NInputOverride] = {
    "acrossfade": _NInputOverride(emit_default=False),
    "ladspa": _NInputOverride(fallback=0, emit_default=False),
}


# The count option's name, in the order to look for it: `inputs` for most
# N-input filters, `nb_inputs` where that is the longer name the registry's
# adjacent-alias dedup keeps (interleave/ainterleave -- `n` is the alias it
# drops). A filter with neither has no count option (`option=None`).
_N_INPUT_OPTION_NAMES = ("inputs", "nb_inputs")


def _n_input_spec(
    name: str, dynamic: DynamicFilter, options: dict[str, FilterOption]
) -> _NInputFilter:
    """One N-input filter's call shape, derived from what this registry reports.

    `stream`/`output` both come from `dynamic.output`: ffmpeg's pad notation
    for an N-input filter is just `N->V`/`N->A`, one letter, and every filter
    observed takes input pads of that same kind. `option`/`fallback` come from
    the filter's own option table; `_N_INPUT_OVERRIDES` covers the two things
    no single build can answer about itself.
    """
    option_name = next((n for n in _N_INPUT_OPTION_NAMES if n in options), None)
    fallback = 2
    emit_default = True
    if option_name is not None:
        default = options[option_name].default
        if default is not None:
            try:
                fallback = int(float(default))
            except ValueError:
                pass
    else:
        fallback = 0
        emit_default = False
    override = _N_INPUT_OVERRIDES.get(name)
    if override is not None:
        if override.fallback is not None:
            fallback = override.fallback
        if override.emit_default is not None:
            emit_default = override.emit_default
    return _NInputFilter(
        name=name,
        stream=dynamic.output,
        output=dynamic.output,
        option=option_name,
        fallback=fallback,
        emit_default=emit_default,
    )


_N_INPUT_HINT = (
    "the number of streams you pass IS the filter's input count; either pass "
    "that many streams, or set the count explicitly, e.g. amix(a, b, c, inputs => 3)"
)


# `concat` stays excluded from the registry (dynamic on the OUTPUT side too,
# `N->N` -- see registry.py), but VARIADIC gives its count a source, so it is
# callable on those terms alone -- never without VARIADIC.
_CONCAT_NAME = "concat"


_CONCAT_VARIADIC_HINT = (
    "concat has a variable pad count: call it with VARIADIC, e.g. "
    "concat(VARIADIC array_agg(v))"
)
