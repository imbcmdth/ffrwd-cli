"""The library half of the MCP server: one function per tool, no SDK import.

Everything a tool does happens here, over the library API -- never over
:mod:`ffrwd.cli`, whose handlers print to stdout, which a stdio server
cannot share with its protocol stream. Nothing in this module writes to
stdout or stderr; results come back as JSON-able dicts.

Rejections raise `FfrwdError`, whose ``str()`` is the line-anchored message
the SDK turns into a failed tool call. :func:`validate_query` is the
deliberate exception: it NEVER raises and returns the error as data
(`FfrwdError.to_dict()`, the shape ``docs/error-schema.json`` pins),
because it is the structured half of the repair loop.

What a compile has to say short of refusing comes back the same way, in a
``warnings`` array beside the answer: a package resolved from the machine-wide
lockfile instead of the project's, or one linked to a directory no lockfile
pins. Each warning is a code, a message, the namespace it is about, an anchor
and a hint. ``validate`` adds the array only when there is one, so a valid
query with nothing to say still answers with an empty object.

Two tools here are about packages rather than queries.
:func:`search_packages` asks the registry's search, ranked the same way the
site ranks; :func:`install_package` downloads a package and writes it to the
store and to a project's lockfile, which is why the server registers it only
when the capability flag allows it.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from .. import binaries, nn, wasm
from .. import packages as packages_module
from .. import registry as registry_module
from ..compiler import (
    classify,
    compile_all,
    compile_table_sql,
    emitted_commands,
)
from ..diagram import render_diagram
from ..emit import build_ffmpeg_commands
from ..errors import ErrorCode, FfrwdError
from ..execute import DEFAULT_TIMEOUT, PlanResult, execute, execute_plan, render_plan
from ..ir import Graph, SinkUnit
from ..processes import ProcessPlan, SidecarProcess
from ..project import (
    LOCKFILE_NAME,
    MANIFEST_NAME,
    PackageSet,
    RegistryEntry,
    discover,
    find_lockfile,
)
from ..prompt import build_system_prompt
from ..table import TableResult, render_csv, render_table
from ..vars import Substitution, referenced, substitute
from ..warnings import OnWarning, WarningLog

__all__ = [
    "STDERR_LIMIT",
    "compile_query",
    "dialect_prompt",
    "explain_query",
    "inspect_query",
    "install_package",
    "list_filters",
    "run_query",
    "search_packages",
    "validate_query",
]

# Longest ffmpeg stderr `run` returns; the tail is what carries the failure.
STDERR_LIMIT = 8000

_TABLE_MESSAGE = "this query has no media destination (no COPY, or every COPY is FORMAT csv)"
_TABLE_HINT = "call inspect for its rows -- only a media COPY ... TO compiles to ffmpeg"

_MEDIA_MESSAGE = "this query has a media COPY, so it has no rows to show"
_MEDIA_HINT = "call compile for its ffmpeg command"

_NO_PATH_MESSAGE = "a sink in this query names no destination path"
_NO_PATH_HINT = "give every COPY a TO '<path>'; TO STDOUT has no file to write"


def _prepare(query: str, variables: dict[str, str] | None) -> Substitution:
    """`query` with its :name references substituted.

    An UNSET reference becomes NULL -- absence, judged by the compile itself
    -- so the check points the other way: a `vars` entry for a name the query
    never references is the rejection, naming what the query does reference.
    """
    supplied = dict(variables or {})
    names = referenced(query)
    unknown = sorted(name for name in supplied if name not in names)
    if unknown:
        written = ", ".join(f"'{name}'" for name in unknown)
        verb = "names a variable" if len(unknown) == 1 else "name variables"
        listed = (
            "it references " + ", ".join(f":{name}" for name in sorted(names))
            if names
            else "it references no variables"
        )
        raise FfrwdError(
            ErrorCode.UNSUPPORTED_SQL,
            f"vars {written} {verb} the query never references",
            line=1,
            col=1,
            hint=f"drop the entry, or fix the query text; {listed}",
        )
    return substitute(query, supplied)


def _packages(project: str | None) -> PackageSet | None:
    """The project `project` names, or None when the caller named none.

    An editor's agent knows which workspace a query belongs to and the server
    does not, so the path is an argument rather than something read out of the
    process's working directory.
    """
    return None if project is None else discover(Path(project))


def _sinks(graphs: list[Graph]) -> list[SinkUnit]:
    return [unit for graph in graphs for unit in graph.sinks]


def _is_table_query(
    text: str, packages: PackageSet | None = None, on_warning: OnWarning | None = None
) -> bool:
    """True if `text` succeeds as a table/csv query.

    A table query compiles through its own lenient pipeline, so "compiles =
    valid" still holds for one ``compile_commands`` rejected.
    """
    try:
        is_table_capable, _has_copy = classify(text, packages=packages, on_warning=on_warning)
    except FfrwdError:
        return False
    if not is_table_capable:
        return False
    try:
        compile_table_sql(text, packages=packages, on_warning=on_warning)
    except FfrwdError:
        return False
    return True


def _reported(warnings: WarningLog) -> list[dict[str, object]]:
    """What the compile had to say, as the tool result's ``warnings`` array.

    stdout is the protocol stream here, so a diagnostic has nowhere to be
    printed: it comes back as data, next to the answer it is about.
    """
    return [warning.to_dict() for warning in warnings.warnings]


def _warnings_only(warnings: WarningLog) -> dict[str, Any]:
    """`validate`'s success answer: empty, unless there is something to say.

    The tool's contract is that a valid query returns nothing; a warning is
    not an error, so what tells the two apart is the ``code`` an error
    carries, which this never has.
    """
    reported = _reported(warnings)
    return {"warnings": reported} if reported else {}


def _table_error() -> FfrwdError:
    return FfrwdError(ErrorCode.UNSUPPORTED_SQL, _TABLE_MESSAGE, line=1, col=1, hint=_TABLE_HINT)


def compile_query(
    query: str, variables: dict[str, str] | None = None, project: str | None = None
) -> dict[str, Any]:
    """The ffmpeg command(s) `query` compiles to, or the process plan.

    A query that reaches a wasm module is several processes joined by pipes,
    not one command: the answer then carries `pipeline` (the shell line) and
    `plan` (the processes, edges and stages) instead of `commands`.
    """
    sub = _prepare(query, variables)
    packages = _packages(project)
    warnings = WarningLog()
    try:
        compiled = compile_all(
            sub.text, packages=packages, on_warning=warnings, unset=sub.unset
        )
    except FfrwdError:
        # A query with no streaming representation fails here; table mode is
        # the fallback, tried only for a query that could BE one. If it is,
        # the refusal names the tool that handles it.
        if _is_table_query(sub.text, packages, warnings):
            raise _table_error() from None
        raise
    graphs = compiled.graphs
    # A bare SELECT compiles, but names no destination -- compile never
    # invents one, so it is the same refusal as above.
    if any(unit.path is None for unit in _sinks(graphs)):
        raise _table_error()

    if compiled.plan is not None:
        return {
            "pipeline": render_plan(compiled.plan, sidecar_argv=wasm.shown_argv),
            "plan": compiled.plan.to_dict(),
            "outputs": [unit.path for unit in _sinks(graphs) if unit.path is not None],
            "warnings": _reported(warnings),
        }

    emitted = emitted_commands(graphs)
    commands: list[list[str]] = []
    for e in emitted:
        commands += build_ffmpeg_commands(e)
    return {
        "commands": commands,
        "filter_complex": [e.filter_complex for e in emitted],
        "outputs": [unit.path for unit in _sinks(graphs) if unit.path is not None],
        # loudnorm2: the first command measures, and the next carries
        # ${FFRWD_LN_*} placeholders only `run` fills in.
        "needs_measurement": any(bool(e.measure_filter_complex) for e in emitted),
        "warnings": _reported(warnings),
    }


def validate_query(
    query: str, variables: dict[str, str] | None = None, project: str | None = None
) -> dict[str, Any]:
    """Empty if `query` compiles, else the typed error object. Never raises.

    A query that compiled with something to say answers with a ``warnings``
    array and no ``code``; an error always carries one.
    """
    sub: Substitution | None = None
    packages: PackageSet | None = None
    compiling = False
    warnings = WarningLog()
    try:
        sub = _prepare(query, variables)
        # Inside the try: a malformed manifest is a rejection like any other,
        # and this tool returns every rejection as data.
        packages = _packages(project)
        compiling = True
        compile_all(sub.text, packages=packages, on_warning=warnings, unset=sub.unset)
    except FfrwdError as err:
        # The table fallback answers only for a query that failed to COMPILE.
        # An unreferenced `vars` name or a malformed manifest failed before
        # that, and is the answer itself.
        if compiling and sub is not None and _is_table_query(sub.text, packages, warnings):
            return _warnings_only(warnings)
        return err.to_dict()
    except Exception as err:  # no input may make a validate call fail
        return FfrwdError(
            ErrorCode.INTERNAL,
            f"internal error while validating ({err.__class__.__name__}: {err})",
            line=1,
            col=1,
            hint="please report this query as a bug",
        ).to_dict()
    return _warnings_only(warnings)


def explain_query(
    query: str,
    variables: dict[str, str] | None = None,
    project: str | None = None,
    *,
    describe: wasm.Describe = wasm.describe,
    invoke: wasm.Invoke = wasm.invoke,
) -> dict[str, Any]:
    """The IR graph `query` compiles to, one per ffmpeg command.

    A query that partitions into processes carries the plan beside its
    graphs: the processes, the typed edges between them, and the stages.
    ``mermaid`` is the same picture as a mermaid flowchart, always present.
    """
    warnings = WarningLog()
    sub = _prepare(query, variables)
    compiled = compile_all(
        sub.text,
        packages=_packages(project),
        on_warning=warnings,
        unset=sub.unset,
        describe=describe,
        invoke=invoke,
    )
    result: dict[str, Any] = {
        "graphs": [graph.to_dict() for graph in compiled.graphs],
        "mermaid": render_diagram(compiled.graphs, compiled.plan),
        "warnings": _reported(warnings),
    }
    if compiled.plan is not None:
        result["plan"] = compiled.plan.to_dict()
    return result


def _row_text(result: TableResult) -> list[list[str]]:
    """Every cell as the text a table prints, through the CSV renderer.

    The renderers are the only public path to a cell's text, and a stream,
    record or array cell has no JSON form of its own.
    """
    return list(csv.reader(io.StringIO(render_csv(result, header=False))))


def inspect_query(
    query: str, variables: dict[str, str] | None = None, project: str | None = None
) -> dict[str, Any]:
    """The rows of a table query: what tracks, chapters, cues or attachments a file has."""
    sub = _prepare(query, variables)
    packages = _packages(project)
    warnings = WarningLog()
    is_table_capable, _has_copy = classify(
        sub.text, packages=packages, on_warning=warnings, unset=sub.unset
    )
    if not is_table_capable:
        raise FfrwdError(
            ErrorCode.UNSUPPORTED_SQL, _MEDIA_MESSAGE, line=1, col=1, hint=_MEDIA_HINT
        )
    sinks = compile_table_sql(sub.text, packages=packages, on_warning=warnings, unset=sub.unset)
    return {
        "warnings": _reported(warnings),
        "results": [
            {
                "columns": list(sink.result.columns),
                "rows": _row_text(sink.result),
                "text": render_table(sink.result),
                "csv": sink.csv,
                "path": sink.path,
            }
            for sink in sinks
        ]
    }


def _matches(needle: str, name: str, doc: str) -> bool:
    return needle in name.lower() or needle in doc.lower()


def list_filters(pattern: str | None = None) -> dict[str, Any]:
    """What the LOCAL ffmpeg reports, optionally narrowed to a substring."""
    registry = registry_module.load()
    needle = (pattern or "").lower()

    filters: list[dict[str, Any]] = []
    for name in registry.names():
        f = registry.get(name)
        if f is None or not _matches(needle, name, f.doc):
            continue
        filters.append(
            {
                "name": name,
                "inputs": list(f.inputs),
                "output": f.output,
                "timeline": f.timeline,
                "doc": f.doc,
            }
        )

    sources: list[dict[str, Any]] = []
    for name in registry.source_names():
        s = registry.get_source(name)
        if s is None or not _matches(needle, name, s.doc):
            continue
        sources.append({"name": name, "output": s.output, "doc": s.doc})

    result: dict[str, Any] = {
        "available": registry.available(),
        "source": registry.source,
        "filters": filters,
        "sources": sources,
    }
    # An exact name also gets that filter's options, the `name => value`
    # surface. They cost an `ffmpeg -help filter=X`, so never for a listing.
    options = registry.options(pattern) if pattern is not None else None
    if options is not None:
        result["options"] = [
            {
                "name": o.name,
                "type": o.type,
                "doc": o.doc,
                "minimum": o.minimum,
                "maximum": o.maximum,
                "default": o.default,
                "constants": list(o.constants),
            }
            for o in options.values()
            if not o.unusable
        ]
    return result


def _tail(text: str) -> str:
    """The last `STDERR_LIMIT` characters of `text`, where a failure is stated."""
    if len(text) <= STDERR_LIMIT:
        return text
    return "[earlier output dropped]\n" + text[-STDERR_LIMIT:]


def _run_plan(
    plan: ProcessPlan, graphs: list[Graph], timeout: float, overwrite: bool,
    warnings: WarningLog,
) -> dict[str, Any]:
    """Run a partitioned query's plan, stage by stage, and report every member.

    A plan binding a model provisions the inference runtime first, silently:
    the server has no progress channel, and a first run simply takes as long
    as the download does.
    """
    if any(
        isinstance(process, SidecarProcess) and process.models
        for process in plan.processes
    ):
        nn.ensure(announce=lambda line: None)
    result: PlanResult = execute_plan(
        plan, sidecar_argv=wasm.sidecar_argv, timeout=timeout, overwrite=overwrite
    )
    return {
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "stages": [
            {
                "index": stage.index,
                "exit_code": stage.exit_code,
                "timed_out": stage.timed_out,
                "members": [
                    {
                        "id": member.id,
                        "argv": member.argv,
                        "exit_code": member.exit_code,
                        "terminated": member.terminated,
                        "stderr": _tail(member.stderr),
                    }
                    for member in stage.members
                ],
            }
            for stage in result.stages
        ],
        # The members that failed on their own; a lost neighbour's pipes close
        # too, so this is shorter than the nonzero exit codes above.
        "failed": [member.id for member in result.failures],
        "outputs": [unit.path for unit in _sinks(graphs) if unit.path is not None],
        "warnings": _reported(warnings),
    }


def run_query(
    query: str,
    variables: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    overwrite: bool = False,
    project: str | None = None,
) -> dict[str, Any]:
    """Compile `query` and run it, writing the files its COPY names.

    One ffmpeg usually; a query that reaches a wasm module runs as the
    process plan its compile carries, and the result reports `stages` of
    `members` instead of `commands`.
    """
    sub = _prepare(query, variables)
    packages = _packages(project)
    warnings = WarningLog()
    is_table_capable, _has_copy = classify(
        sub.text, packages=packages, on_warning=warnings, unset=sub.unset
    )
    if is_table_capable:
        raise _table_error()

    compiled = compile_all(sub.text, packages=packages, on_warning=warnings, unset=sub.unset)
    graphs = compiled.graphs
    if any(unit.path is None for unit in _sinks(graphs)):
        raise FfrwdError(
            ErrorCode.UNSUPPORTED_SQL, _NO_PATH_MESSAGE, line=1, col=1, hint=_NO_PATH_HINT
        )
    if binaries.ffmpeg_path() is None:
        raise FfrwdError(
            ErrorCode.INTERNAL, "ffmpeg not found", line=1, col=1, hint=binaries.INSTALL_HINT
        )

    if compiled.plan is not None:
        return _run_plan(compiled.plan, graphs, timeout, overwrite, warnings)

    emitted = emitted_commands(graphs)
    # capture_stderr: a server owns no terminal for ffmpeg's progress lines.
    result = execute(emitted, timeout=timeout, overwrite=overwrite, capture_stderr=True)
    return {
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "measure_error": result.measure_error,
        "commands": [
            {"argv": c.argv, "exit_code": c.exit_code, "stderr": _tail(c.stderr)}
            for c in result.commands
        ],
        "outputs": [unit.path for unit in _sinks(graphs) if unit.path is not None],
        "warnings": _reported(warnings),
    }


def search_packages(term: str | None = None) -> dict[str, Any]:
    """What the registry ranks for `term`. Reads the network, writes nothing."""
    return {
        "registry": packages_module.base_url(),
        "packages": [listing.to_dict() for listing in packages_module.search(term)],
    }


def install_package(package: str, project: str) -> dict[str, Any]:
    """Install `package` into the project at `project`: the store, then its lockfile.

    Fetches what it depends on too, recursively, each at its highest
    published version -- see `brought`.

    `project` is required and is never created: a directory with no lockfile
    at or above it is not a project, and inventing one is not this tool's
    call.
    """
    lock = find_lockfile(Path(project))
    if lock is None:
        raise FfrwdError(
            ErrorCode.UNSUPPORTED_SQL,
            f"no {LOCKFILE_NAME} in {project} or above it",
            hint="run `ffrwd init` in that directory first; installing never creates one",
        )
    manifest = lock.parent / MANIFEST_NAME
    installed = packages_module.install(
        package, lock=lock, manifest=manifest if manifest.is_file() else None
    )
    replaced = installed.replaced
    return {
        "registry": packages_module.base_url(),
        "name": installed.release.name,
        "version": installed.release.version,
        "brought": [
            {"name": one.name, "version": one.version} for one in installed.brought
        ],
        "sha256": installed.release.sha256,
        "downloaded": installed.downloaded,
        "lockfile": str(installed.lock),
        "manifest": None if installed.manifest is None else str(installed.manifest),
        "replaced": None
        if replaced is None
        else replaced.name
        if isinstance(replaced, RegistryEntry)
        else replaced.path,
    }


def dialect_prompt() -> str:
    """The system prompt describing the dialect, for this machine's ffmpeg."""
    return build_system_prompt(registry_module.load())
