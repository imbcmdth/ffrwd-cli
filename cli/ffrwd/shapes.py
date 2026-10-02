"""A node module's shape for one call: its ports, how each pairs, its clock.

A module exporting ``ffrwd:av@0.19.1``'s ``node`` says what it reads and
writes per call, not once: its ports and formats turn on its params and on
which inputs the call binds. The compiler asks the sidecar for each distinct
call, ``ffrwd-wasm --shape <module> --params <json> --bound <bindings>``,
which prints the WIT's ``node-shape`` record as JSON (:class:`NodeShape`).
The bound inputs go as JSON, one :class:`Binding` per input with a
:class:`StreamHint` per stream, so a shape may turn on the clock's rate.
:func:`shape` runs that, and like :func:`ffrwd.wasm.describe` it is a seam: a
lowering test hands over its own and nothing is spawned.

A shape is a pure function of the module's bytes, the params and the bound
list, hints and all, so :class:`ShapeCache` asks once per distinct triple.

Also here, since every reader of a shape needs them: the streaming words a
window is said in (:func:`window_words`), and structural row matching
(:func:`row_mismatch`), which compares the JSON schemas two data ports carry.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Literal, cast

from . import binaries
from .errors import ErrorCode, FfrwdError

__all__ = [
    "Accepts",
    "Anchor",
    "Binding",
    "Clock",
    "Hold",
    "InputPort",
    "Interval",
    "NodeShape",
    "OutputFormat",
    "OutputPort",
    "Pairing",
    "Shape",
    "ShapeCache",
    "StreamHint",
    "bound_json",
    "node_shape",
    "row_mismatch",
    "shape",
    "window_words",
]

PortKind = Literal["video", "audio", "data", "packets"]
RowsUse = Literal["ignore", "per-frame", "state"]
PairingKind = Literal["lockstep", "hold", "interval", "arrival"]
AnchorKind = Literal["shared-clock", "first-frame", "tagged"]
ClockKind = Literal["input", "rate", "rate-of", "self-clocked"]
FormatKind = Literal["video", "audio", "data", "packets", "like"]
Wants = Literal["all", "keyframes", "first", "timing"]

_PORT_KINDS: tuple[PortKind, ...] = ("video", "audio", "data", "packets")
_ROWS_USES: tuple[RowsUse, ...] = ("ignore", "per-frame", "state")
_WANTS: tuple[Wants, ...] = ("all", "keyframes", "first", "timing")
_ANCHORS: tuple[AnchorKind, ...] = ("shared-clock", "first-frame", "tagged")

_SHAPE_FLAG = "--shape"
_PARAMS_FLAG = "--params"
_PARAMS_FROM_FLAG = "--params-from"
_BOUND_FLAG = "--bound"

# Params longer than this go in a file rather than on the command line, which
# Windows caps at 32,767 characters for everything on it.
PARAMS_INLINE_LIMIT = 4096

_UNKNOWN_HINT = "the module may be built against a sidecar this ffrwd does not know"


@dataclass(frozen=True)
class Anchor:
    """How a hold or interval input's offset is fixed; `tag` names the tag
    `tagged` reads."""

    kind: AnchorKind
    tag: str = ""


_SHARED_CLOCK = Anchor("shared-clock")


@dataclass(frozen=True)
class Hold:
    anchor: Anchor
    lead: float
    linger: float | None = None
    timeout: float | None = None
    group: str | None = None
    port_param: str | None = None


@dataclass(frozen=True)
class Interval:
    latency: float | None = None
    ahead: float = 0.0
    anchor: Anchor = _SHARED_CLOCK
    group: str | None = None

    @property
    def retimed(self) -> bool:
        """Whether the host re-stamps the stream onto the clock: its pts count
        from an origin of their own, not the clock's."""
        return self.anchor.kind != "shared-clock"


@dataclass(frozen=True)
class Pairing:
    kind: PairingKind
    hold: Hold | None = None
    interval: Interval | None = None


@dataclass(frozen=True)
class Accepts:
    pixel_formats: tuple[str, ...] = ()
    sample_formats: tuple[str, ...] = ()
    sample_rates: tuple[int, ...] = ()
    channel_counts: tuple[int, ...] = ()
    codecs: tuple[str, ...] = ()
    wants: Wants = "all"
    like: str | None = None


@dataclass(frozen=True)
class InputPort:
    name: str
    kind: PortKind
    required: bool
    many: bool
    pairing: Pairing
    rows: RowsUse
    window: int
    stride: int
    accepts: Accepts
    schema: Mapping[str, object] | None = None


@dataclass(frozen=True)
class OutputFormat:
    """One arm of ``output-format``.

    `video`: width, height, `pixel_format`. `audio`: `sample_rate`,
    `channels`, `sample_format`. `data`: `codec`. `packets`: the coded
    stream, `codec`, `time_base`, what it carries as `coded` (video, audio or
    data) with that kind's own fields, and `extradata` as hex. `like`:
    `port`, with `pixel_format` or `sample_format` the field it overrides.
    """

    kind: FormatKind
    width: int | None = None
    height: int | None = None
    pixel_format: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    sample_format: str | None = None
    codec: str | None = None
    time_base: tuple[int, int] | None = None
    port: str | None = None
    coded: Literal["video", "audio", "data"] | None = None
    extradata: str = ""


@dataclass(frozen=True)
class OutputPort:
    name: str
    kind: PortKind
    format: OutputFormat | None = None
    time_base: tuple[int, int] | None = None
    latency: float = 0.0
    schema: Mapping[str, object] | None = None
    row: int | None = None


@dataclass(frozen=True)
class Clock:
    """`port` for `input` and `rate-of`, `rate` for `rate`."""

    kind: ClockKind
    port: str = ""
    rate: tuple[int, int] | None = None


@dataclass(frozen=True)
class NodeShape:
    inputs: tuple[InputPort, ...]
    outputs: tuple[OutputPort, ...]
    clock: Clock
    pure: bool = True
    one_to_one: bool = False
    bounded: bool = True
    relation: tuple[Mapping[str, object], ...] = ()
    # The JSON the sidecar printed, kept for `explain` and the IR.
    raw: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)

    def input(self, name: str) -> InputPort | None:
        return next((port for port in self.inputs if port.name == name), None)

    def output(self, name: str) -> OutputPort | None:
        return next((port for port in self.outputs if port.name == name), None)

    @property
    def clock_input(self) -> InputPort | None:
        """The input the clock names, for an input clock; None otherwise."""
        return self.input(self.clock.port) if self.clock.kind == "input" else None

    def to_dict(self) -> dict[str, object]:
        return dict(self.raw)


@dataclass(frozen=True)
class StreamHint:
    """What the compiler knows of one bound stream before the run: a video
    stream's frame rate or an audio stream's sample rate, None where nothing
    settles it."""

    rate: Fraction | None = None


@dataclass(frozen=True)
class Binding:
    """One input a call binds, as `shape` is told it: a hint per stream, in
    the order the call names them."""

    input: str
    streams: tuple[StreamHint, ...] = (StreamHint(),)


def bound_json(bound: Sequence[Binding]) -> str:
    """`bound` as ``--bound`` takes it."""
    return json.dumps(
        [
            {
                "input": binding.input,
                "streams": [
                    {
                        "rate": None
                        if hint.rate is None
                        else {"num": hint.rate.numerator, "den": hint.rate.denominator}
                    }
                    for hint in binding.streams
                ],
            }
            for binding in bound
        ],
        separators=(",", ":"),
    )


def _reject(message: str, hint: str = _UNKNOWN_HINT) -> FfrwdError:
    return FfrwdError(ErrorCode.UNSUPPORTED_SQL, message, hint=hint)


# -- reading the JSON -------------------------------------------------------


def _object(value: object, what: str, module: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise _reject(f"the sidecar's shape of {module} has {what} that is not an object")
    return cast(Mapping[str, object], value)


def _kind(value: Mapping[str, object], what: str, module: str) -> str:
    kind = value.get("kind")
    if not isinstance(kind, str):
        raise _reject(f"the sidecar's shape of {module} has {what} with no kind")
    return kind


def _arm(value: Mapping[str, object], kind: str) -> object:
    """A variant's payload: nested under its kind's own name, or the object itself."""
    nested = value.get(kind.replace("-", "_"))
    if nested is None:
        nested = value.get("value")
    return value if nested is None else nested


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _whole(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _texts(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _wholes(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(found for item in value if (found := _whole(item)) is not None)


def _rational(value: object) -> tuple[int, int] | None:
    if isinstance(value, dict):
        num, den = _whole(value.get("num")), _whole(value.get("den"))
    elif isinstance(value, list) and len(value) == 2:
        num, den = _whole(value[0]), _whole(value[1])
    else:
        return None
    if num is None or den is None or den == 0:
        return None
    return (num, den)


def _schema(value: object) -> Mapping[str, object] | None:
    """A port's JSON schema: written as a string of JSON, or as the object."""
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return cast(Mapping[str, object], value) if isinstance(value, dict) else None


def _choice(value: object, choices: tuple[str, ...], fallback: str) -> str:
    if isinstance(value, str):
        spelled = value.replace("_", "-")
        if spelled in choices:
            return spelled
    return fallback


def _anchor(value: object, module: str) -> Anchor:
    if isinstance(value, str):
        kind = _choice(value, ("shared-clock", "first-frame"), "first-frame")
        return Anchor(cast(AnchorKind, kind))
    raw = _object(value, "a hold anchor", module)
    kind = _choice(_kind(raw, "a hold anchor", module), _ANCHORS, "")
    if not kind:
        raise _reject(f"the sidecar's shape of {module} names an anchor this ffrwd does not know")
    tag = ""
    if kind == "tagged":
        arm = _arm(raw, kind)
        tag = _text(arm) or (_text(raw.get("name")) or _text(raw.get("tag")) or "")
    return Anchor(cast(AnchorKind, kind), tag)


def _pairing(value: object, module: str, port: str) -> Pairing:
    what = f"input '{port}''s pairing"
    if isinstance(value, str):
        kind = _choice(value, ("lockstep", "arrival"), "")
        if not kind:
            raise _reject(f"the sidecar's shape of {module} gives {what} no fields")
        return Pairing(cast(PairingKind, kind))
    raw = _object(value, what, module)
    kind = _choice(_kind(raw, what, module), ("lockstep", "hold", "interval", "arrival"), "")
    if kind == "hold":
        arm = _object(_arm(raw, kind), what, module)
        return Pairing(
            "hold",
            hold=Hold(
                anchor=_anchor(arm.get("anchor"), module),
                lead=_number(arm.get("lead")) or 0.0,
                linger=_number(arm.get("linger")),
                timeout=_number(arm.get("timeout")),
                group=_text(arm.get("group")),
                port_param=_text(arm.get("port_param")),
            ),
        )
    if kind == "interval":
        arm = _object(_arm(raw, kind), what, module)
        anchor = arm.get("anchor")
        return Pairing(
            "interval",
            interval=Interval(
                latency=_number(arm.get("latency")),
                ahead=_number(arm.get("ahead")) or 0.0,
                anchor=_SHARED_CLOCK if anchor is None else _anchor(anchor, module),
                group=_text(arm.get("group")),
            ),
        )
    if kind in ("lockstep", "arrival"):
        return Pairing(cast(PairingKind, kind))
    raise _reject(
        f"the sidecar's shape of {module} pairs input '{port}' in a way this ffrwd does not know"
    )


def _accepts(value: object) -> Accepts:
    if not isinstance(value, dict):
        return Accepts()
    return Accepts(
        pixel_formats=_texts(value.get("pixel_formats")),
        sample_formats=_texts(value.get("sample_formats")),
        sample_rates=_wholes(value.get("sample_rates")),
        channel_counts=_wholes(value.get("channel_counts")),
        codecs=_texts(value.get("codecs")),
        wants=cast(Wants, _choice(value.get("wants"), _WANTS, "all")),
        like=_text(value.get("like")),
    )


def _port_kind(value: object, module: str, port: str) -> PortKind:
    kind = _choice(value, _PORT_KINDS, "")
    if not kind:
        raise _reject(
            f"the sidecar's shape of {module} gives port '{port}' a kind this ffrwd does not know"
        )
    return cast(PortKind, kind)


def _name(raw: Mapping[str, object], what: str, module: str) -> str:
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise _reject(f"the sidecar's shape of {module} has {what} with no name")
    return name


def _input_port(value: object, module: str) -> InputPort:
    raw = _object(value, "an input port", module)
    name = _name(raw, "an input port", module)
    window = _whole(raw.get("window")) or 1
    stride = _whole(raw.get("stride")) or 1
    return InputPort(
        name=name,
        kind=_port_kind(raw.get("kind"), module, name),
        required=raw.get("required") is True,
        many=raw.get("many") is True,
        pairing=_pairing(raw.get("pairing"), module, name),
        rows=cast(RowsUse, _choice(raw.get("rows"), _ROWS_USES, "ignore")),
        window=max(window, 1),
        stride=max(stride, 1),
        accepts=_accepts(raw.get("accepts")),
        schema=_schema(raw.get("schema")),
    )


def _output_format(value: object, module: str, port: str) -> OutputFormat | None:
    if value is None:
        return None
    what = f"output '{port}''s format"
    raw = _object(value, what, module)
    kind = _choice(_kind(raw, what, module), ("video", "audio", "data", "packets", "like"), "")
    arm = _arm(raw, kind)
    if kind == "data":
        codec = _text(arm) if not isinstance(arm, dict) else _text(arm.get("codec"))
        return OutputFormat("data", codec=codec or "json")
    body = _object(arm, what, module)
    if kind == "video":
        return OutputFormat(
            "video",
            width=_whole(body.get("width")),
            height=_whole(body.get("height")),
            pixel_format=_text(body.get("pix_fmt")) or _text(body.get("pixel_format")),
        )
    if kind == "audio":
        return OutputFormat(
            "audio",
            sample_rate=_whole(body.get("sample_rate")),
            channels=_whole(body.get("channels")),
            sample_format=_text(body.get("sample_fmt")) or _text(body.get("sample_format")),
        )
    if kind == "packets":
        coded = body.get("format")
        carried = coded if isinstance(coded, dict) else {}
        coded_kind = _choice(carried.get("kind"), ("video", "audio", "data"), "")
        return OutputFormat(
            "packets",
            codec=_text(body.get("codec")),
            time_base=_rational(body.get("time_base")),
            coded=cast(Literal["video", "audio", "data"], coded_kind) if coded_kind else None,
            width=_whole(carried.get("width")),
            height=_whole(carried.get("height")),
            sample_rate=_whole(carried.get("sample_rate")),
            channels=_whole(carried.get("channels")),
            extradata=_text(body.get("extradata")) or "",
        )
    if kind == "like":
        return OutputFormat(
            "like",
            port=_text(body.get("port")),
            pixel_format=_text(body.get("pixel_format")),
            sample_format=_text(body.get("sample_format")),
        )
    raise _reject(
        f"the sidecar's shape of {module} gives output '{port}' a format this ffrwd does not know"
    )


def _output_port(value: object, module: str) -> OutputPort:
    raw = _object(value, "an output port", module)
    name = _name(raw, "an output port", module)
    return OutputPort(
        name=name,
        kind=_port_kind(raw.get("kind"), module, name),
        format=_output_format(raw.get("format"), module, name),
        time_base=_rational(raw.get("time_base")),
        latency=max(_number(raw.get("latency")) or 0.0, 0.0),
        schema=_schema(raw.get("schema")),
        row=_whole(raw.get("row")),
    )


def _clock(value: object, module: str) -> Clock:
    if isinstance(value, str):
        if value.replace("_", "-") == "self-clocked":
            return Clock("self-clocked")
        raise _reject(f"the sidecar's shape of {module} names a clock with no fields")
    raw = _object(value, "a clock", module)
    kind = _choice(_kind(raw, "a clock", module), ("input", "rate", "rate-of", "self-clocked"), "")
    arm = _arm(raw, kind)
    if kind in ("input", "rate-of"):
        port = _text(arm) if not isinstance(arm, dict) else _text(arm.get("port"))
        if not port:
            raise _reject(f"the sidecar's shape of {module} names a clock with no input")
        return Clock(cast(ClockKind, kind), port=port)
    if kind == "rate":
        nested = isinstance(arm, dict) and "num" not in arm
        rate = _rational(arm.get("rate") if nested and isinstance(arm, dict) else arm)
        if rate is None or rate[0] <= 0 or rate[1] <= 0:
            raise _reject(f"the sidecar's shape of {module} names a rate clock with no rate")
        return Clock("rate", rate=rate)
    if kind == "self-clocked":
        return Clock("self-clocked")
    raise _reject(f"the sidecar's shape of {module} names a clock this ffrwd does not know")


def _relation(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        return ()
    rows: list[Mapping[str, object]] = []
    for item in value:
        found = _schema(item)
        rows.append(found if found is not None else {})
    return tuple(rows)


def node_shape(module: str, payload: object) -> NodeShape:
    """One ``--shape`` document as a :class:`NodeShape`, or a rejection naming `module`."""
    raw = _object(payload, "a document", module)
    inputs = raw.get("inputs")
    outputs = raw.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise _reject(f"the sidecar's shape of {module} names no inputs or no outputs")
    return NodeShape(
        inputs=tuple(_input_port(port, module) for port in inputs),
        outputs=tuple(_output_port(port, module) for port in outputs),
        clock=_clock(raw.get("clock"), module),
        pure=raw.get("pure") is not False,
        one_to_one=raw.get("one_to_one") is True,
        bounded=raw.get("bounded") is not False,
        relation=_relation(raw.get("relation")),
        raw=dict(raw),
    )


# -- asking the sidecar -----------------------------------------------------


# Runs one module's `shape` for one call: :func:`shape` is the real one, and a
# lowering test passes its own. `grants` are the sidecar flags a module's own
# imports need to answer at all (a source reading the network for its
# outputs), ahead of the flag that dispatches the call.
Shape = Callable[[str, str, Sequence[Binding], Sequence[str]], NodeShape]


def shape(
    module: str, params: str, bound: Sequence[Binding], grants: Sequence[str] = ()
) -> NodeShape:
    """Ask the sidecar for the shape of `module` under `params` with `bound` bound.

    Raises ``FfrwdError`` and nothing else, unanchored: the caller anchors it
    on the call that named the module.
    """
    from .wasm import (  # deferred: wasm imports processes, which reads shapes
        INSTALL_HINT,
        _first_line,
        timeout_seconds,
    )

    sidecar = binaries.ffrwd_wasm_path()
    if sidecar is None:
        raise _reject(
            f"the ffrwd-wasm sidecar is not installed, and the shape of '{module}' "
            "needs it to read the module",
            hint=INSTALL_HINT,
        )
    argv = [sidecar, *grants, _SHAPE_FLAG, module]
    if bound:
        argv += [_BOUND_FLAG, bound_json(bound)]
    budget = timeout_seconds()
    with tempfile.TemporaryDirectory(prefix="ffrwd-shape-") as scratch:
        if len(params) > PARAMS_INLINE_LIMIT:
            written = Path(scratch) / "params.json"
            written.write_text(params, encoding="utf-8")
            argv += [_PARAMS_FROM_FLAG, str(written)]
        elif params and params != "{}":
            argv += [_PARAMS_FLAG, params]
        done = _run_shape(sidecar, module, argv, budget)
    if done.returncode != 0:
        raise _reject(
            f"the module '{module}' refused the shape for these params: "
            f"{_first_line(done.stderr)}",
            hint="check the arguments match what the module declares",
        )
    try:
        payload = json.loads(done.stdout)
    except ValueError as err:
        raise _reject(
            f"the ffrwd-wasm sidecar's shape of {module} is not JSON",
            hint="the sidecar on PATH may be a different version than this ffrwd",
        ) from err
    return node_shape(module, payload)


def _run_shape(
    sidecar: str, module: str, argv: list[str], budget: float
) -> subprocess.CompletedProcess[str]:
    """One ``--shape`` run, refused by name when it cannot be run at all."""
    from .wasm import INSTALL_HINT, _budget_hint  # deferred: wasm imports processes

    try:
        return subprocess.run(
            argv,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=budget,
            check=False,
        )
    except (OSError, ValueError) as err:
        raise _reject(
            f"could not run the ffrwd-wasm sidecar at {sidecar}: "
            f"{getattr(err, 'strerror', None) or err}",
            hint=INSTALL_HINT,
        ) from err
    except subprocess.TimeoutExpired as err:
        raise _reject(
            f"the ffrwd-wasm sidecar did not shape {module} within {budget:g}s",
            hint=_budget_hint(budget),
        ) from err


def _module_hash(module: str) -> str:
    """The module file's own digest, or its path where the file cannot be read.

    A path that names nothing readable still shapes the same within one
    compile, which is all a test's fake module needs.
    """
    try:
        return hashlib.sha256(Path(module).read_bytes()).hexdigest()
    except OSError:
        return f"path:{module}"


class ShapeCache:
    """One :data:`Shape` asked once per (module bytes, params, bound list)."""

    def __init__(self, ask: Shape = shape) -> None:
        self._ask = ask
        self._hashes: dict[str, str] = {}
        self._shapes: dict[tuple[str, str, tuple[Binding, ...]], NodeShape] = {}

    def __call__(
        self, module: str, params: str, bound: Sequence[Binding], grants: Sequence[str] = ()
    ) -> NodeShape:
        digest = self._hashes.get(module)
        if digest is None:
            digest = self._hashes[module] = _module_hash(module)
        key = (digest, params, tuple(bound))
        found = self._shapes.get(key)
        if found is None:
            found = self._shapes[key] = self._ask(module, params, bound, grants)
        return found


# -- windows, in streaming words -------------------------------------------


def window_words(port: InputPort, rate: Fraction | None = None) -> str:
    """The clock input's window as streaming SQL says it.

    `rate` is items per second (frames, or samples for audio); with it the
    window is said in seconds, without it in items.
    """
    if port.window <= 1:
        return "per-frame"

    def span(items: int) -> str:
        if rate is None or rate <= 0:
            unit = "samples" if port.kind == "audio" else "frames"
            return f"{items} {unit}"
        return f"{_seconds(Fraction(items) / rate)} s"

    if port.stride >= port.window:
        return f"tumbling {span(port.window)}"
    if port.stride == 1 and port.kind != "audio":
        return f"sliding {span(port.window)}"
    return f"hopping {span(port.window)} every {span(port.stride)}"


def _seconds(value: Fraction | float) -> str:
    """Seconds as a reader writes them: 2, 0.5, 0.033."""
    rounded = round(float(value), 3)
    return f"{rounded:g}"


# -- structural row matching ------------------------------------------------


def _types(schema: Mapping[str, object]) -> frozenset[str] | None:
    written = schema.get("type")
    if isinstance(written, str):
        return frozenset({written})
    if isinstance(written, list):
        return frozenset(item for item in written if isinstance(item, str))
    return None


def _covers(reader: frozenset[str], producer: frozenset[str]) -> bool:
    """Whether every type the producer may write is one the reader takes.

    JSON Schema's integer is a number, so a reader of numbers takes a
    producer of integers; the reverse is not so.
    """
    return all(t in reader or (t == "integer" and "number" in reader) for t in producer)


def _type_text(types: frozenset[str] | None) -> str:
    if not types:
        return "no type"
    return " or ".join(sorted(types))


def row_mismatch(
    reader: Mapping[str, object], producer: Mapping[str, object]
) -> tuple[str, str, str] | None:
    """The first field `reader` names that `producer` lacks or types otherwise.

    ``(field, what the reader wants, what the producer writes)``, or None
    where every field the reader names is there with a type it takes. Fields
    the producer writes beyond those pass. Nested objects and arrays are
    compared the same way, a nested field named by its dotted path.
    """
    wanted = reader.get("properties")
    if not isinstance(wanted, dict):
        return None
    written = producer.get("properties")
    given: Mapping[str, object] = written if isinstance(written, dict) else {}
    for name, want in wanted.items():
        if not isinstance(want, dict):
            continue
        have = given.get(name)
        want_types = _types(want)
        if not isinstance(have, dict):
            return (name, _type_text(want_types), "nothing")
        have_types = _types(have)
        if want_types is not None and (have_types is None or not _covers(want_types, have_types)):
            return (name, _type_text(want_types), _type_text(have_types))
        nested = row_mismatch(want, have)
        if nested is not None:
            return (f"{name}.{nested[0]}", nested[1], nested[2])
        items_want, items_have = want.get("items"), have.get("items")
        if isinstance(items_want, dict) and isinstance(items_have, dict):
            item_types_want, item_types_have = _types(items_want), _types(items_have)
            if item_types_want is not None and (
                item_types_have is None or not _covers(item_types_want, item_types_have)
            ):
                return (
                    f"{name}[]",
                    _type_text(item_types_want),
                    _type_text(item_types_have),
                )
    return None
