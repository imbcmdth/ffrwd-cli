"""The MCP SDK wiring: tool and resource registration, and the stdio loop.

The only module that imports ``mcp``. Every tool body is in
:mod:`ffrwd.mcp.tools`; what is here is the argument plumbing and the text
a model reads to choose a tool.

stdout belongs to the protocol. Nothing in ffrwd prints -- the library
raises `FfrwdError` where the CLI would print, and no diagnostic has any
other channel -- and while ``stdio_server`` is serving it points file
descriptor 1 at stderr and serves the wire from a private duplicate, so a
stray write from any library or child process misses the protocol stream.
``run``'s ffmpeg children inherit that redirected descriptor, and their
stderr is captured into the tool result rather than written anywhere.

One capability flag, not a matrix. ``allow_unsafe`` is the whole of it: the
tools that only answer -- about a query, or about what the registry publishes
-- are always there, and the ones that do something are behind it: ``run``,
which writes files on model say-so, and ``install``, which downloads code and
writes it to disk and to a project's lockfile. A permissions matrix for a
local dev tool invites passing every flag, and the per-call prompting already
lives in the MCP client. So the flag is a coarse capability switch, and the
precision that matters goes in each tool's DESCRIPTION, since that is the text
a client shows when it asks.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .. import __version__
from ..errors import FfrwdError
from ..execute import DEFAULT_TIMEOUT
from . import tools

__all__ = ["build_server", "serve"]

DIALECT_URI = "ffrwd://dialect"

_INSTRUCTIONS = """\
ffrwd compiles Postgres-dialect SQL into ffmpeg commands: a FROM is an
input file, a column is a stream, a function call is a filter, and COPY ... TO
is the output file.

Read the ffrwd://dialect resource before writing a query -- it is the whole
grammar, generated for this machine's ffmpeg. When a query is rejected, call
validate to get the typed error (code, line, col, message, hint), fix what it
names, and validate again. Call filters to check that a filter and its options
exist in the local ffmpeg build rather than assuming. search finds installable
packages in the registry; pass `project` so a query can call what a project
installed.\
"""


# One function per tool below, named FOR the tool: the SDK reads each tool's
# name, argument schema and description straight off the function, so a
# rename here renames the tool.


def compile(
    query: str, vars: dict[str, str] | None = None, project: str | None = None
) -> dict[str, Any]:
    """Compile a query into the ffmpeg command(s) it runs as, without running them.

    Returns `commands` (each an argv list, run in order), `filter_complex`
    (the graph string per command), `outputs` (the files the query would
    write), and `needs_measurement` (true when the first command measures for
    the next, whose ${FFRWD_LN_*} placeholders only the run tool fills in).

    A query that reaches a wasm module is several processes joined by pipes,
    not one command: the answer then carries `pipeline` (the shell line) and
    `plan` (the processes, edges and stages) instead of `commands`.

    `vars` supplies :name / :'name' / :"name" substitutions.

    `project` is the directory (or query file path) the query belongs to;
    ffrwd walks up from it for a `ffrwd.json` and a `ffrwd.lock`, and
    makes the namespaced functions of that project and of everything it (or
    this machine) installed callable. Omit it for a query that stands alone.
    """
    return tools.compile_query(query, vars, project)


def validate(
    query: str, vars: dict[str, str] | None = None, project: str | None = None
) -> dict[str, Any]:
    """Check that a query compiles; the repair loop's tool.

    Returns an empty object when the query is valid, otherwise the error:
    `code` (the error kind), `line` and `col` (where in the query, 1-based,
    or null), `message`, and `hint` (how to fix it, or null). Never fails --
    an invalid query is a result, not an error. A valid query that resolved a
    package worth mentioning answers with only a `warnings` array, which has
    no `code`: that is what tells a warning from an error.

    `vars` supplies :name / :'name' / :"name" substitutions.

    `project` is the directory (or query file path) the query belongs to;
    ffrwd walks up from it for a `ffrwd.json` and a `ffrwd.lock`, and
    makes the namespaced functions of that project and of everything it (or
    this machine) installed callable. Omit it for a query that stands alone.
    """
    return tools.validate_query(query, vars, project)


def explain(
    query: str, vars: dict[str, str] | None = None, project: str | None = None
) -> dict[str, Any]:
    """The compiled filter graph as JSON: nodes, edges, inputs and sinks.

    One graph per ffmpeg command. Use it to see which filter a column
    expression became and how the streams were wired. `mermaid` in the result
    is the same picture as a mermaid flowchart, ready to render or show.

    `vars` supplies :name / :'name' / :"name" substitutions.

    `project` is the directory (or query file path) the query belongs to;
    ffrwd walks up from it for a `ffrwd.json` and a `ffrwd.lock`, and
    makes the namespaced functions of that project and of everything it (or
    this machine) installed callable. Omit it for a query that stands alone.
    """
    return tools.explain_query(query, vars, project)


def inspect(
    query: str, vars: dict[str, str] | None = None, project: str | None = None
) -> dict[str, Any]:
    """Run a query that returns rows: what is inside a media file.

    For a query with no media COPY -- a bare SELECT, or a COPY ... WITH
    (FORMAT csv) -- over a file's tracks, chapters, cues or attachments. Reads
    the file, writes nothing. Each result has `columns`, `rows`, and `text`
    (the same rows as a printable table).

    `vars` supplies :name / :'name' / :"name" substitutions.

    `project` is the directory (or query file path) the query belongs to;
    ffrwd walks up from it for a `ffrwd.json` and a `ffrwd.lock`, and
    makes the namespaced functions of that project and of everything it (or
    this machine) installed callable. Omit it for a query that stands alone.
    """
    return tools.inspect_query(query, vars, project)


def filters(pattern: str | None = None) -> dict[str, Any]:
    """The filters and sources the LOCAL ffmpeg build has, not what is typical.

    `pattern` is a case-insensitive substring matched against each name and
    its one-line description; omit it for everything. When `pattern` is
    exactly one filter's name, the result also carries that filter's
    `options` -- the `name => value` arguments it accepts, with types, ranges
    and enum constants.
    """
    return tools.list_filters(pattern)


def search(term: str | None = None) -> dict[str, Any]:
    """Find installable ffrwd packages in the package registry.

    The registry ranks `term` and answers most relevant first: an exact
    package name, then a near name, then the words of each package's
    description, keywords, function names and recipe names. Omit it to
    browse everything published. A term matching nothing returns an empty
    list.

    Each package has a `name` (`<namespace>/<package>`, what the install tool
    takes and what a query's calls are qualified by), its latest `version`, a
    `description`, the `functions` and `recipes` it provides, and
    `installs_week`, how often it was installed over the trailing week.

    Reads the registry over the network and writes nothing; a registry that
    cannot be reached is a typed error, never a silently empty list.
    `registry` is the base it read.
    """
    return tools.search_packages(term)


def install(package: str, project: str) -> dict[str, Any]:
    """Install a package from the registry. This DOWNLOADS CODE and WRITES FILES.

    It fetches an archive over the network, unpacks it into this machine's
    package store under the home directory, and edits two files in `project`:
    it pins the package in `ffrwd.lock` and records it as a dependency in
    `ffrwd.json`. The code it installs becomes callable by every query
    compiled in that project.

    It also walks the package's own manifest for what IT depends on, and
    installs each of those the same way, recursively -- each at its highest
    published version, pinned in the lockfile but not the manifest, which
    only ever names what was asked for. `brought` in the result lists what
    came along.

    A package that pins models fetches those too, from Hugging Face, each
    verified against the sha256 its manifest records -- model files can be
    hundreds of megabytes, so this can take a while.

    The archive is verified against the sha256 the registry publishes before
    it is opened, so a download that does not match is discarded and nothing
    is written -- but the registry's own contents are not reviewed by anything
    here. Install only what the user asked for by name.

    `package` is `<namespace>/<package>`, or `<namespace>/<package>@<version>`
    for an exact version; without one, the highest published version is
    installed and pinned exactly.

    `project` is the directory the project lives in, and is required: a
    directory with no `ffrwd.lock` at or above it is not a project, and this
    tool never creates one.
    """
    return tools.install_package(package, project)


def run(
    query: str,
    vars: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    overwrite: bool = False,
    project: str | None = None,
) -> dict[str, Any]:
    """Compile a query and execute ffmpeg. This WRITES FILES on disk.

    Every path the query's COPY ... TO names is created or, with `overwrite`,
    replaced -- this tool is the only one here that changes anything outside
    the answer it returns.

    Returns the run's `exit_code` (0 on success), `timed_out`, and one entry
    per command with its `argv`, `exit_code` and captured `stderr` (tail only
    when long). `outputs` lists the files written. `timeout` is per command,
    in seconds. `overwrite` false makes ffmpeg refuse to replace an existing
    file.

    A query that reaches a wasm module runs as several processes joined by
    pipes; the result then reports `stages` of `members` instead of
    `commands`, and a first run of a query that infers may first download the
    inference runtime, which takes as long as it takes.

    `vars` supplies :name / :'name' / :"name" substitutions.

    `project` is the directory (or query file path) the query belongs to;
    ffrwd walks up from it for a `ffrwd.json` and a `ffrwd.lock`, and
    makes the namespaced functions of that project and of everything it (or
    this machine) installed callable. Omit it for a query that stands alone.
    """
    return tools.run_query(query, vars, timeout, overwrite, project)


def _surfaced(tool: Callable[..., Any]) -> Callable[..., Any]:
    """The typed message crosses the SDK boundary.

    The SDK masks an arbitrary exception's text in the client-visible error
    ("Error executing tool ..."); a rejection's line-anchored message IS this
    server's contract -- the repair loop reads it -- so it is re-raised as the
    SDK's own ToolError, whose text survives.
    """

    @functools.wraps(tool)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return tool(*args, **kwargs)
        except FfrwdError as err:
            raise ToolError(str(err)) from err

    return wrapped


def build_server(*, allow_unsafe: bool = False) -> MCPServer[Any]:
    """The configured server; `allow_unsafe` adds the tools that write: ``run`` and ``install``."""
    # log_level configures the root logger, and at INFO sqlglot narrates every
    # array subscript it rewrites -- a line per query in the client's log pane.
    server: MCPServer[Any] = MCPServer(
        "ffrwd", version=__version__, instructions=_INSTRUCTIONS, log_level="WARNING"
    )
    for tool in (compile, validate, explain, inspect, filters, search):
        server.add_tool(_surfaced(tool))
    if allow_unsafe:
        server.add_tool(_surfaced(run))
        server.add_tool(_surfaced(install))

    @server.resource(
        DIALECT_URI,
        name="ffrwd dialect",
        description=(
            "The ffrwd SQL dialect: grammar, the functions this machine's "
            "ffmpeg provides, worked examples, and what each error code means."
        ),
        mime_type="text/plain",
    )
    def _dialect() -> str:
        return tools.dialect_prompt()

    return server


def serve(*, allow_unsafe: bool = False) -> None:
    """Serve over stdin/stdout until the client disconnects."""
    build_server(allow_unsafe=allow_unsafe).run("stdio")
