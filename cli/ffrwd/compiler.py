"""The compiler pipeline: SQL text in, IR :class:`~ffrwd.ir.Graph` out.

``compile_sql`` chains the first three passes plus the PTS-reset and split
passes::

    parse -> resolve -> probe -> lower -> insert_pts_resets -> insert_splits

The returned graph is split-complete — every pad has exactly one consumer,
which is what :func:`ffrwd.emit.emit` expects.
:func:`ffrwd.pts.insert_pts_resets` runs first so the split pass sees the
final topology, including any reset node it added.

Probing happens between resolve and lower: every
distinct input path is probed exactly once, results re-keyed by ALIAS for
:func:`ffrwd.lower.lower`, so two aliases over one file share a probe.
:func:`ffrwd.probe.probe` never raises and returns ``None`` for a URL, a
missing file, or a missing/failing ffprobe, so an unreadable input never
blocks a compile — it only loses subscript-bound validation and provenance
metadata. Probing is unconditional; there is no ``probe=False``.

The filter REGISTRY is the whole function surface: every call name
resolves in it and nowhere else. :func:`ffrwd.registry.load` never raises
and degrades to an empty registry when ffmpeg is missing, so a compile without
ffmpeg is not an error — it is one where every call name is UNKNOWN_FUNCTION.

Every entry point takes three optional keywords beyond the SQL: `packages`,
the :class:`~ffrwd.project.PackageSet` a namespaced call resolves in;
`on_warning`, the callback that hears what a compile has to say short of
refusing it (:mod:`ffrwd.warnings`); and `unset`, variable substitution's
map from an unset variable's NULL (its (line, col) in the text) to the
variable's name (:class:`ffrwd.vars.Substitution`), which is what lets a
rejection at the NULL's point of use name the variable. All default to None.

Guardrail #7 lives here: no input, however malformed, may produce anything but
a compile result or a :class:`~ffrwd.errors.FfrwdError`. Each pass carries
its own backstop; this one catches the rest (recursion limits, sqlglot
internals) as ``INTERNAL``, the code the fuzz corpus asserts never fires.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from . import registry as registry_module
from . import wasm
from .emit import Emitted, emit
from .errors import ErrorCode, FfrwdError
from .execute import DEFAULT_TIMEOUT
from .functions import WasmFunction
from .inputs import forces_demuxer, probe_options, render_options
from .ir import Graph
from .lower import input_option_values, lower_commands, lower_table
from .parser import Resolved, parse, resolve
from .probe import ProbeFailure, ProbeResult
from .probe import probe as probe_path
from .probe import probe_failure as probe_failure_path
from .processes import (
    AudioFormat,
    ModelBinding,
    ModuleShape,
    ProcessPlan,
    external_filters,
    partition,
)
from .project import PackageSet
from .pts import insert_pts_resets
from .split import insert_splits
from .table import TableSink
from .warnings import OnWarning
from .wasm import Described

__all__ = [
    "Compiled",
    "classify",
    "compile_all",
    "compile_commands",
    "compile_sql",
    "compile_table_sql",
    "emitted_commands",
    # The probe seam: what a caller holding no media file replaces, so that a
    # query naming one still compiles. `probe_failure_path` is its
    # companion: the detail behind a `None` from `probe_path` for a path
    # that DOES exist.
    "probe_failure_path",
    "probe_path",
]


def emitted_commands(graphs: Sequence[Graph]) -> list[Emitted]:
    """Every ffmpeg command a compile's graphs hold, in order.

    A COPY writing a module's ROWS has no ffmpeg command at all: the sidecar
    writes that file, so the graph carries no sink and there is nothing to
    render. Every other graph emits exactly as it always has.
    """
    return [emit(graph) for graph in graphs if graph.sinks]


def _probe_flags(res: Resolved, alias: str) -> tuple[tuple[str, ...], bool]:
    """This alias's ffprobe argv, and whether the full option set forces the demuxer.

    Only the options :func:`ffrwd.inputs.probe_options` keeps reach ffprobe --
    ffmpeg-only flags like ``realtime`` or ``itsoffset`` shape decode, not
    what a probe would read, and several of them ffprobe rejects outright.
    ``forces_demuxer`` still reads the FULL set: a forced format always
    probes (it has to, to be read at all), so the two agree regardless.

    Best effort: an option lowering will reject comes back as no flags at
    all, because probing runs first and never raises. The rejection still
    lands moments later, anchored on the option itself.
    """
    raw = res.input_options.get(alias)
    if not raw:
        return (), False
    try:
        values = input_option_values(raw)
    except FfrwdError:
        return (), False
    return tuple(render_options(probe_options(values))), forces_demuxer(values)


def _probe_inputs(
    res: Resolved,
) -> tuple[dict[str, ProbeResult | None], dict[str, ProbeFailure | None]]:
    """Probe each distinct (path, options) once; results and failures keyed by alias.

    Options are part of the key: one path read two ways -- a different
    demuxer, a different capture size -- is two inputs to ffmpeg and two
    probes here. Two aliases spelling the same path AND the same options
    still share one probe.

    The failure dict holds an entry only where the probe came back ``None``
    for a path :func:`ffrwd.probe.probe_failure` has something to say about
    -- a path that exists but that ffprobe itself declined to read. It says
    WHY, for an honest rejection later instead of a second guess at the same
    question a missing file would also answer with ``None``.
    """
    by_input: dict[tuple[str, tuple[str, ...]], ProbeResult | None] = {}
    by_input_failure: dict[tuple[str, tuple[str, ...]], ProbeFailure | None] = {}
    by_alias: dict[str, ProbeResult | None] = {}
    by_alias_failure: dict[str, ProbeFailure | None] = {}
    for alias, index in res.sources.items():
        path = res.input_paths[index]
        flags, forced = _probe_flags(res, alias)
        key = (path, flags)
        if key not in by_input:
            by_input[key] = probe_path(path, flags, forced_format=forced)
            by_input_failure[key] = (
                None
                if by_input[key] is not None
                else probe_failure_path(path, flags, forced_format=forced)
            )
        by_alias[alias] = by_input[key]
        by_alias_failure[alias] = by_input_failure[key]
    return by_alias, by_alias_failure


def _describe_modules(
    res: Resolved, describe: wasm.Describe = wasm.describe
) -> dict[str, Described]:
    """Describe each distinct wasm module once; return the results keyed by path.

    A describe that fails is a REJECTION, unlike a probe: the module's pixel
    format and parameters are what the processes around it are built from, so
    there is nothing to fall back to. The rejection is re-anchored on the
    declaration that named the module, which carries its own position.

    `describe` is a parameter so a test can compile a query naming a module
    without a sidecar on the machine.
    """
    described: dict[str, Described] = {}
    for declared in res.wasm.values():
        if declared.module in described:
            continue
        try:
            described[declared.module] = describe(declared.module)
        except FfrwdError as err:
            raise FfrwdError(
                err.code,
                f"function '{declared.name}': {err.message}",
                line=declared.line,
                col=declared.col,
                hint=err.hint,
            ) from err
    return described


def _stream_wasm(res: Resolved) -> dict[str, WasmFunction]:
    """The declared wasm functions that filter a STREAM, keyed by name.

    A value-returning one is folded at compile time and never becomes a
    sidecar process, so it takes no part in partitioning: not the process
    plan, not the pixel-format negotiation, not the external filter list.
    """
    return {name: declared for name, declared in res.wasm.items() if not declared.is_value}


def _wire_formats(
    declared_stream: Mapping[str, WasmFunction], describes: Mapping[str, Described]
) -> dict[str, str]:
    """The pixel format each VIDEO module's edges carry, keyed by module path."""
    formats: dict[str, str] = {}
    for declared in _negotiable(declared_stream, describes, "video", formats):
        described = describes[declared.module]
        with _at(declared):
            formats[declared.module] = wasm.wire_pix_fmt(described)
    return formats


def _audio_wires(
    declared_stream: Mapping[str, WasmFunction], describes: Mapping[str, Described]
) -> dict[str, AudioFormat]:
    """What each AUDIO module's edges carry, keyed by module path.

    The pcm the negotiated sample format travels as, and the rate and channel
    count the module wants the stream conformed to.
    """
    wires: dict[str, AudioFormat] = {}
    for declared in _negotiable(declared_stream, describes, "audio", wires):
        described = describes[declared.module]
        with _at(declared):
            wires[declared.module] = wasm.wire_audio(described)
    return wires


def _negotiable(
    declared_stream: Mapping[str, WasmFunction],
    describes: Mapping[str, Described],
    kind: str,
    done: Mapping[str, object],
) -> list[WasmFunction]:
    """The declarations of `kind` whose module still needs a wire format.

    Keyed on the DECLARED kind rather than the described one: lowering already
    refused a declaration whose module filters the other kind, and a module
    that named no format at all belongs to whichever negotiation then reports
    it having named none.
    """
    found: list[WasmFunction] = []
    seen: set[str] = set()
    for declared in declared_stream.values():
        if declared.module in done or declared.module in seen:
            continue
        if declared.module not in describes or declared.stream_kind != kind:
            continue
        if describes[declared.module].packet_sink:
            continue  # its edge carries encoded packets, not frames
        seen.add(declared.module)
        found.append(declared)
    return found


@contextmanager
def _at(declared: WasmFunction) -> Iterator[None]:
    """Re-anchor a wire-format rejection on the declaration that named it."""
    try:
        yield
    except FfrwdError as err:
        raise FfrwdError(
            err.code,
            f"function '{declared.name}': {err.message}",
            line=declared.line,
            col=declared.col,
            hint=err.hint,
        ) from err


def _nn_models(
    declared_stream: Mapping[str, WasmFunction], describes: Mapping[str, Described]
) -> dict[str, ModelBinding]:
    """The model each module that runs one binds, keyed by module path.

    A module whose describe says it runs none binds nothing and emits no
    ``-nn``. One that does names the file at compile time, so a missing model
    is a rejection here rather than a load failure at run time.
    """
    models: dict[str, ModelBinding] = {}
    for declared in declared_stream.values():
        described = describes.get(declared.module)
        if described is None or not described.nn or declared.module in models:
            continue
        with _at(declared):
            models[declared.module] = wasm.model_binding(described, declared.module)
    return models


def _effect_grants(
    declared_stream: Mapping[str, WasmFunction], describes: Mapping[str, Described]
) -> dict[str, tuple[str, ...]]:
    """What each module needs granted, keyed by module path.

    Read off the describe: a module importing wasi:http needs ``http``, one
    importing wasi:sockets needs ``udp``. The sidecar denies both without
    the matching argv, which is what these become.
    """
    found: dict[str, tuple[str, ...]] = {}
    for declared in declared_stream.values():
        described = describes.get(declared.module)
        if described is None or declared.module in found:
            continue
        effects = tuple(
            effect
            for effect, needed in (("http", described.http), ("udp", described.udp))
            if needed
        )
        if effects:
            found[declared.module] = effects
    return found


def _module_shapes(
    declared_stream: Mapping[str, WasmFunction], describes: Mapping[str, Described]
) -> dict[str, ModuleShape]:
    """What each module declares about its frame timing, keyed by module path."""
    return {
        declared.module: describes[declared.module].shape
        for declared in declared_stream.values()
        if declared.module in describes
    }


def _anchored(
    err: FfrwdError, declared_stream: Mapping[str, WasmFunction]
) -> FfrwdError:
    """A partition rejection re-anchored on the declaration that named the module.

    Partitioning knows a module by its path and nothing about the text; the
    declaration carries the position, so the rejection is rewritten to name
    the function and point at it.
    """
    if err.line is not None:
        return err
    for declared in declared_stream.values():
        if declared.module in err.message:
            return FfrwdError(
                err.code,
                f"function '{declared.name}': {err.message}",
                line=declared.line,
                col=declared.col,
                hint=err.hint,
            )
    return err


@dataclass(frozen=True)
class Compiled:
    """What one compile produced: the commands, and the plan that runs them.

    `plan` is None for a query one ffmpeg runs on its own, which is every
    query that names no wasm module -- the caller then emits `graphs` the way
    it always has. It is a :class:`~ffrwd.processes.ProcessPlan` when a module
    IS named, since ffmpeg cannot host one and the streams around it move
    between processes over pipes.

    `default_timeout` is what a run applies when the caller sets no timeout of
    its own: seconds scaled to the longest input, or None -- no timeout --
    when any input's duration is unknown (see :func:`_default_timeout`).
    """

    graphs: list[Graph]
    plan: ProcessPlan | None = None
    default_timeout: float | None = None


def _default_timeout(probes: Mapping[str, ProbeResult | None]) -> float | None:
    """The timeout a run of this compile applies when the caller sets none.

    Ten times the longest input duration, floored at ``DEFAULT_TIMEOUT``: a
    slow encode legitimately runs at many multiples of its material's length.
    None -- no timeout at all -- when any input's duration is unknown (a
    device, a live URL, a lavfi graph, an unreadable file), since nothing
    bounds how long such a run should live.
    """
    # A no-forward-progress watchdog eventually replaces the duration
    # multiple; until then the multiple is the budget.
    durations: list[float] = []
    for result in probes.values():
        if result is None or result.duration is None:
            return None
        durations.append(result.duration)
    if not durations:
        return None
    return max(float(DEFAULT_TIMEOUT), 10 * max(durations))


def compile_sql(
    text: str,
    *,
    packages: PackageSet | None = None,
    on_warning: OnWarning | None = None,
    owner: tuple[str, str] | None = None,
    unset: Mapping[tuple[int, int], str] | None = None,
) -> Graph:
    """Compile SQL `text` into a split-complete IR graph.

    The FIRST command's graph, which is the whole query except for the one
    fan-out shape that compiles to a command sequence;
    :func:`compile_commands` returns every command's.

    Every input is probed opportunistically (see module docstring). The
    installed ffmpeg's filter set IS the function surface, so what
    compiles depends on what that ffmpeg reports; tests wanting a fixed,
    machine-independent surface call :func:`ffrwd.lower.lower` directly with
    a registry built from the captured snapshot.

    Raises ``FfrwdError`` — and nothing else — on every rejection.
    """
    return compile_commands(text, packages=packages, on_warning=on_warning, unset=unset)[0]


def compile_commands(
    text: str,
    *,
    packages: PackageSet | None = None,
    on_warning: OnWarning | None = None,
    owner: tuple[str, str] | None = None,
    unset: Mapping[tuple[int, int], str] | None = None,
) -> list[Graph]:
    """Compile SQL `text` into one split-complete IR graph per ffmpeg COMMAND.

    Usually one graph, a fan-out ``COPY ... TO (<expression>)`` included: its
    files become sink units of a single graph, one ffmpeg command with several
    outputs. The exception is a fan-out that trims and stream-copies every
    stream it maps, which stays one graph per file. Same probing and registry
    contract as :func:`compile_sql`.

    Raises ``FfrwdError`` — and nothing else — on every rejection.
    """
    return compile_all(
        text, packages=packages, on_warning=on_warning, owner=owner, unset=unset
    ).graphs


def compile_all(
    text: str,
    *,
    packages: PackageSet | None = None,
    on_warning: OnWarning | None = None,
    owner: tuple[str, str] | None = None,
    unset: Mapping[tuple[int, int], str] | None = None,
    describe: wasm.Describe = wasm.describe,
    invoke: wasm.Invoke = wasm.invoke,
) -> Compiled:
    """Compile SQL `text` into its commands, and the plan that runs them.

    :func:`compile_commands` without the part that throws the plan away. A
    query naming a ``LANGUAGE wasm`` module compiles to a
    :class:`~ffrwd.processes.ProcessPlan`, because ffmpeg cannot host the
    module and the streams around it travel between processes; every other
    query leaves ``plan`` None and is the single ffmpeg command it always was.

    `describe` is how a module is read, a parameter for the same reason
    lowering's `describes` is: a caller with no sidecar can still compile.
    `invoke` is the same for a VALUE-returning module: lowering runs it once
    per call site to fold the result, and a test hands over its own so
    folding spawns nothing.

    Raises ``FfrwdError`` — and nothing else — on every rejection.
    """
    try:
        res = resolve(parse(text, unset), packages=packages, on_warning=on_warning, owner=owner)
        probes, probe_failures = _probe_inputs(res)
        describes = _describe_modules(res, describe)
        graphs = lower_commands(
            res,
            probes,
            registry=registry_module.load(),
            on_warning=on_warning,
            describes=describes,
            invoke=invoke,
            probe_failures=probe_failures,
        )
        ready = [insert_splits(insert_pts_resets(graph)) for graph in graphs]
        budget = _default_timeout(probes)
        stream_wasm = _stream_wasm(res)
        if not stream_wasm:
            return Compiled(graphs=ready, default_timeout=budget)
        try:
            plan = partition(
                ready[0],
                external=external_filters(
                    *sorted({d.module for d in stream_wasm.values()})
                ),
                probes=probes,
                pix_fmts=_wire_formats(stream_wasm, describes),
                shapes=_module_shapes(stream_wasm, describes),
                audio_wires=_audio_wires(stream_wasm, describes),
                models=_nn_models(stream_wasm, describes),
                effects=_effect_grants(stream_wasm, describes),
                anchors=res.input_anchors,
            )
        except FfrwdError as err:
            raise _anchored(err, stream_wasm) from err
        return Compiled(graphs=ready, plan=plan, default_timeout=budget)
    except FfrwdError:
        raise
    except RecursionError as err:
        raise FfrwdError(
            ErrorCode.UNSUPPORTED_SQL,
            "query nests too deeply to compile",
            line=1,
            col=1,
            hint="flatten the expression: fewer nested parentheses or calls",
        ) from err
    except Exception as err:  # guardrail #7: no panics on user input
        raise FfrwdError(
            ErrorCode.INTERNAL,
            f"internal error while compiling ({err.__class__.__name__}: {err})",
            line=1,
            col=1,
            hint="please report this query as a bug",
        ) from err


def classify(
    text: str,
    *,
    packages: PackageSet | None = None,
    on_warning: OnWarning | None = None,
    owner: tuple[str, str] | None = None,
    unset: Mapping[tuple[int, int], str] | None = None,
) -> tuple[bool, bool]:
    """``(is_table_capable, has_copy)`` for `text`.

    Cheap and static: parse + resolve only, no probing. ``is_table_capable``
    is True when `text` has no media destination -- a bare SELECT, or every
    COPY a ``FORMAT csv`` one. A bare SELECT is always table-capable by this
    check alone; the CLI decides from it whether to use
    :func:`compile_table_sql` or fall back to :func:`compile_sql`.

    Raises ``FfrwdError`` on a query that does not even resolve.
    """
    res = resolve(parse(text, unset), packages=packages, on_warning=on_warning, owner=owner)
    return all(sink.is_csv for sink in res.sinks), bool(res.sinks)


def compile_table_sql(
    text: str,
    *,
    packages: PackageSet | None = None,
    on_warning: OnWarning | None = None,
    owner: tuple[str, str] | None = None,
    unset: Mapping[tuple[int, int], str] | None = None,
) -> list[TableSink]:
    """Compile SQL `text` into its printable table/csv result set(s).

    The sibling of :func:`compile_sql` for a table query: one
    :class:`~ffrwd.table.TableSink` per COPY, or one for a bare SELECT.
    Metadata columns and NULL-row gaps, both rejections under
    :func:`compile_sql`, are legal here — that is the whole difference. Inputs
    are probed opportunistically, same as :func:`compile_sql`.

    Raises ``FfrwdError`` — and nothing else — on every rejection.
    """
    try:
        res = resolve(parse(text, unset), packages=packages, on_warning=on_warning, owner=owner)
        probes, probe_failures = _probe_inputs(res)
        return lower_table(
            res,
            probes,
            registry=registry_module.load(),
            on_warning=on_warning,
            probe_failures=probe_failures,
        )
    except FfrwdError:
        raise
    except RecursionError as err:
        raise FfrwdError(
            ErrorCode.UNSUPPORTED_SQL,
            "query nests too deeply to compile",
            line=1,
            col=1,
            hint="flatten the expression: fewer nested parentheses or calls",
        ) from err
    except Exception as err:  # guardrail #7: no panics on user input
        raise FfrwdError(
            ErrorCode.INTERNAL,
            f"internal error while compiling ({err.__class__.__name__}: {err})",
            line=1,
            col=1,
            hint="please report this query as a bug",
        ) from err
