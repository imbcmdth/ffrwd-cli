"""Command-line interface for ffrwd.

Thin wrapper around the library pipeline (``compile_sql`` -> ``emit`` ->
``build_ffmpeg_commands``). See the "CLI" section of ffrwd-project.md.

A compile is a SEQUENCE of ffmpeg commands — one for every query but a
``two_pass`` sink or a ``ffrwd.loudnorm2`` graph (two each), and a
stream-copied fan-out with trim windows (one per output file, the only
form ffmpeg cuts copied streams correctly in).
``compile`` prints them joined by `` && `` on one line; ``run`` hands them to
``ffrwd.execute``, which runs them in order, stops at the first nonzero exit
and reports it, with the timeout applied per command. That timeout is
``--timeout`` when one is given, and otherwise the compile's own budget --
ten times the longest input's duration, at least 600s, and none at all when
any input's duration is unknown, which is every live or device input.

``loudnorm2`` is the one compile whose printed line is not pure ffmpeg: its
measuring pass is wrapped in ``eval "$(... | ffrwd loudnorm2env)"``, which
makes the printed form POSIX-shell only. ``run`` needs no shell — it captures
the measuring pass's stderr, parses it in process (``ffrwd.loudnorm``) and
substitutes the numbers straight into the second command's argv.

Subcommands:

* ``compile SQL [-f FILE] [--graph-only]`` -- print the full ffmpeg command
  (POSIX-quoted via ``shlex.join`` even on Windows: it is documentation
  output, not something to paste into cmd.exe), or just the
  ``-filter_complex`` string with ``--graph-only``. The query names its own
  destination with ``COPY ... TO``; a query with no media ``COPY`` -- a bare
  SELECT, or one whose every ``COPY`` is ``FORMAT csv`` -- has nothing to
  compile and is refused (run it instead).
* ``explain SQL [-f FILE] [--mermaid | --diagram]`` -- dump the IR graph as
  JSON, sinks included. ``--mermaid`` prints it as a mermaid flowchart
  instead; ``--diagram`` renders that flowchart in the terminal, which needs
  the optional ``diagram`` extra -- without it, one stderr line naming the
  install command and exit 1, the same shape as ``mcp``'s missing extra.
* ``validate SQL [-f FILE] [--json]`` -- exit 0 silent on success; on error,
  exit 1 with a one-line human message or ``err.to_dict()`` JSON.
* ``run SQL [-f FILE] [--timeout SECS] [-y]`` -- compile and execute ffmpeg
  as a subprocess (guardrail #6: argv list, no shell, timeout enforced,
  stderr surfaced on failure). A query with no media ``COPY`` prints its
  result set as a table (or CSV, for ``COPY ... WITH (FORMAT csv)``);
  otherwise it runs the compiled ffmpeg command(s) against the ``COPY``'s
  own destination paths. ``--remote`` submits the run to the hosted runner
  instead (``ffrwd.remote``): file inputs are hashed and uploaded, "://"
  inputs pass through for the runner to open, and no local ffmpeg is needed
  -- ``ffrwd jobs`` follows the job from there.
* ``jobs [--json | --watch | --cancel ID | --fetch ID [-y]]`` -- the runs
  submitted with ``--remote``: list them (with this month's usage), watch
  the listing until nothing is running, ask for a cancellation, or download
  a succeeded job's outputs to their as-written paths. An ID may be any
  unique prefix of the id ``submit`` printed.
* ``list [--json]`` -- print what the project at the working directory and its
  dependencies provide: one table of packages, one of the functions they
  export, one of the recipes they ship with the variables each declares, and
  one of the dependencies each manifest declares. Takes no query; the export
  list is the manifest's ``lib``, with parameter types read from the files.
* ``init [--name NAME] [--namespace NS] [--rust]`` -- write ``ffrwd.json``, an
  empty ``ffrwd.lock`` and a starter recipe into the working directory. The
  package segment is the directory's name unless ``--name`` says otherwise;
  the namespace is ``--namespace``'s, or derived from the git remote's owner,
  or required. Refuses to overwrite any file it would write. ``--rust``
  scaffolds a wasm module package instead of the bare one: a cargo crate
  whose ``build.rs`` finds the wit, an ``invert`` module in Rust, the lib SQL
  declaring it and a recipe calling it. ``cargo build --target wasm32-wasip2
  --release`` then ``ffrwd publish`` is the whole path from there.
* ``search [TERM] [--json]`` -- ask the registry what it ranks for TERM and
  print it, most relevant first. No term browses everything; a term matching
  nothing is an empty table, exit 0.
* ``install PKG[@VERSION] [-g]`` -- resolve a package by fetching its detail
  document, verify the archive it publishes against the digest it records,
  put it in the store, fetch the models it pins, and pin it in the lockfile
  and the manifest. No version means
  the highest published one, written exact. Then walks its own manifest's
  dependencies and installs each the same way, recursively, at its highest
  published version -- only the package named on the command line is
  recorded in the manifest, what came along is the lockfile's own. Same
  project rule as ``link``: outside a project and without ``-g``, exit 2.
  With no package at all, installs the project standing here: the
  dependencies its own manifest pins, at their written versions, plus its
  pinned models and the runtime its modules load -- a fresh clone builds
  and publishes after this.
* ``path PKG [-g]`` -- print where an installed package's content is on this
  machine, resolved through the same lockfile ``install`` writes: the store
  directory for an installed package, the linked directory for a link. One
  line, the path alone, because a build script reads it -- a module's
  ``build.rs`` asks ``ffrwd path ffrwd/wasm`` for the wit. A package the
  lockfile does not pin is a typed error naming ``ffrwd install``.
* ``link PATH [-g]`` / ``unlink NAME [-g]`` -- record (or drop) a package
  read live out of a directory, in this project's lockfile or, with ``-g``,
  the machine-wide one. The entry records only the directory; the package's
  name comes from the manifest there. Outside a project and without ``-g``
  there is no lockfile to write and none is invented: usage error, exit 2.
* ``login --token TOKEN`` / ``logout`` -- save (or remove) the token this
  machine publishes with. ``login`` checks the token's shape and touches no
  network; whether it is live is the registry's answer at the first command
  that uses it.
* ``publish`` -- validate the package at or above the working directory --
  its manifest, its exports, what its modules declare, every recipe compiled
  against a synthetic file, and every dependency resolved in the registry --
  then pack it and upload it in one request. A refusal from the registry is
  printed as its own message and hint. The manifest's ``private`` key decides
  each version's visibility.
* ``prompt`` -- print the LLM system prompt to stdout. Takes no arguments and
  touches no files, but calls ``registry.load()`` to render the filter
  reference from this machine's ``ffmpeg -filters``/``-help`` output.
* ``mcp [--allow-unsafe]`` -- serve the compiler to an editor or agent as a
  stdio MCP server (``ffrwd.mcp``). Takes no arguments and needs the
  optional ``mcp`` extra; without it, one stderr line naming the install
  command and exit 1. stdout carries the protocol from the moment it starts,
  so nothing else may write there -- ``--allow-unsafe`` adds the tools that
  do something other than answer about a query.
* ``loudnorm2env`` -- read ffmpeg's stderr on stdin, print the
  ``export FFRWD_LN_*=`` lines its loudnorm JSON block holds. Takes no
  arguments and touches no files; exit 1 with one stderr line if there is no
  such block. It exists for the printed ``loudnorm2`` command line, which
  pipes pass 1 into it.

``compile``/``explain``/``validate``/``run`` take the query as SQL TEXT on
the command line. ``-f/--file PATH`` reads it from a file instead (``-f -``
reads stdin, e.g. for the LLM repair loop's pipe). Exactly one of the two is
required; both or neither is a usage error, exit 2. If the positional string
fails to compile and looks like a filename, a stderr hint suggests ``-f``
(see ``_maybe_print_file_hint``).

The positional may also name a RECIPE a package ships (``ffrwd run
split-chapters -v source=film.mkv``). One rule decides which it is, in
``_resolve_query`` so every one of the four subcommands reads it the same
way: text beginning with ``SELECT``, ``COPY``, ``CREATE`` or ``WITH`` is SQL,
always; anything else that matches a recipe's name is that recipe's file;
anything else is SQL and fails as it always did. ``ns.pkg.recipe`` names one
of a map ``bin``'s entries, ``ns.pkg`` a string ``bin``'s recipe; either says
which package when a bare name matches more than one.

A compile can also have something to say short of refusing: a call that
resolved to a machine-wide package rather than to one this project installed,
or a call into a linked directory, which no lockfile pins. Those print as
``warning:`` lines on stderr after the command has run -- never on stdout,
which carries the ffmpeg command, the IR JSON or ``validate --json``'s error
object -- and never twice for one package.

Long steps narrate on stderr: one present-tense line per meaningful step --
an archive fetched, a model downloaded, an input uploaded -- with names and
sizes, and an ASCII spinner while a step with nothing finer to say runs (a
compile, the registry, the job service). The spinner shows only when stderr
is a TTY, so pipes and CI logs carry the lines alone; ``-q/--quiet`` keeps
only the final result. stdout carries neither.

Two flags are deliberately absent. ``--no-probe`` made a READABLE
file compile as if unreadable, silently stripping provenance metadata -- a
determinism switch that changed the result; opportunistic probing already
degrades on unreadable inputs. ``--portable`` had no portable
subset left to mean anything against: every function is a filter of the
installed ffmpeg, so the ffmpeg build answers "will this compile elsewhere".
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

from . import binaries, credentials, diagram, loudnorm, nn, remote, show, store, wasm
from . import packages as packages_module
from . import publish as publish_module
from . import registry as registry_module
from .compiler import (
    Compiled,
    classify,
    compile_all,
    compile_commands,
    compile_table_sql,
    emitted_commands,
)
from .console import Console
from .emit import Emitted, build_ffmpeg_commands
from .errors import ErrorCode, FfrwdError
from .execute import DEFAULT_TIMEOUT, execute, execute_plan, render_plan
from .functions import Signature, package_signatures
from .ir import Graph, SinkUnit
from .probe import is_url
from .processes import ProcessPlan, SidecarProcess
from .project import (
    LOCKFILE_NAME,
    MANIFEST_NAME,
    README_NAME,
    RESERVED_NAMESPACES,
    STATEMENT_KEYWORDS,
    LinkEntry,
    LockEntry,
    Package,
    PackageSet,
    discover,
    entry_root,
    find_lockfile,
    find_manifest,
    held_entry,
    is_namespace,
    is_package_name,
    name_refusal,
    read_lockfile,
    read_manifest,
    stored_name,
    with_entry,
    without_entry,
    write_lockfile,
    write_manifest,
)
from .prompt import build_system_prompt
from .publish import required_variables as _required_variables
from .split import insert_splits
from .table import CellValue, TableResult, TableSink, render_csv, render_table
from .vars import Variable, declared_variables, referenced, substitute, unset_variable
from .warnings import OnWarning, WarningLog

__all__ = ["main"]

# `compile` prints a command SEQUENCE as one line: shell chaining, so the
# printed line runs the passes in order when pasted.
_CHAIN = " && "

# `run` is the DEFAULT subcommand, unconditionally: any argv whose
# first token is not one of these names IS run's argv, flags included
# (`ffrwd -f q.sql`). No plausibility checking -- a mistyped subcommand falls
# through to run's SQL parser and dies as a line-anchored PARSE_ERROR, a
# better diagnostic than a usage line. Consequence: `ffrwd -h` shows run's
# help, not the top-level one.
_SUBCOMMANDS = frozenset(
    {
        "compile",
        "explain",
        "validate",
        "run",
        "jobs",
        "list",
        "init",
        "search",
        "install",
        "path",
        "link",
        "unlink",
        "login",
        "logout",
        "publish",
        "setup",
        "prompt",
        "mcp",
        loudnorm.ENV_SUBCOMMAND,
    }
)


def _version() -> str:
    return metadata.version("ffrwd")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    # psql's spelling (-v is taken by variables there too); checked before the
    # run dispatch, which would otherwise hand the flag to the SQL parser.
    if argv and argv[0] in ("--version", "-V"):
        print(f"ffrwd {_version()}")
        return 0
    if not argv or argv[0] not in _SUBCOMMANDS:
        argv = ["run", *argv]

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_usage(sys.stderr)
        return 2

    handler = _HANDLERS[args.command]
    # One sink per invocation: `compile` and `validate` compile the same text
    # twice (the table-query fallback), and the reader wants each warning once.
    warnings = WarningLog()
    try:
        return handler(args, warnings)
    finally:
        _print_warnings(warnings)


_QUERY_HELP = "SQL query text (exactly one of this or -f/--file is required)"
_FILE_HELP = "read the query from a file instead of the command line ('-' for stdin)"
_SET_HELP = "define a variable for :name/:'name'/:\"name\" substitution (repeatable)"


def _add_query_arguments(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("query", nargs="?", default=None, help=_QUERY_HELP)
    subparser.add_argument("-f", "--file", default=None, help=_FILE_HELP)
    subparser.add_argument(
        "-v",
        "--set",
        dest="set_vars",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=_SET_HELP,
    )


def _add_quiet_argument(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="print only the final result: no narration, no spinner",
    )


_JOBS_HELP = (
    "host this many instances of each wasm module at once (default: 1); a "
    "module that carries state between calls keeps one"
)


def _add_jobs_argument(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--jobs", type=int, default=1, metavar="N", help=_JOBS_HELP)


def _check_jobs(args: argparse.Namespace) -> int:
    """0 for a usable ``--jobs``, or 2 with the usage error printed."""
    jobs = int(getattr(args, "jobs", 1))
    if jobs >= 1:
        return 0
    print(
        f"error: {args.command}: --jobs must be 1 or more, got {jobs}",
        file=sys.stderr,
    )
    print(
        "hint: --jobs 1 hosts one instance of each module, which is what a "
        "run with no --jobs does",
        file=sys.stderr,
    )
    return 2


# Why a region keeps one worker, by the property that decided it.
_SERIAL_REASONS: Mapping[str, str] = {
    "state": "a module that carries state between calls is hosted by one instance",
    "network": "modules wired to each other hand frames on in order",
    "packets": "a module reading encoded packets takes them in decode order",
}


def _serial_modules(plan: ProcessPlan) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """What `plan` hosts serially, as ``(reason, modules)``.

    A region keeps one worker for exactly one of three reasons, and which one
    is what the run is owed: an impure module is the module author's to
    change, a network of them is the query's, and a packet sink is neither.
    """
    found: dict[str, list[str]] = {name: [] for name in _SERIAL_REASONS}
    for process in plan.processes:
        if not isinstance(process, SidecarProcess) or process.parallel:
            continue
        if process.impure:
            reason, modules = "state", list(process.impure)
        else:
            reason = "packets" if process.packet_sink else "network"
            modules = [process.module]
        found[reason] += [one for one in modules if one not in found[reason]]
    return tuple((reason, tuple(named)) for reason, named in found.items() if named)


def _print_jobs_notice(plan: ProcessPlan | None, jobs: int) -> None:
    """Name what a ``--jobs`` above 1 did not reach, once per reason, on stderr.

    A run that asked for parallel hosting and got none of it somewhere is
    owed the reason; a run that got all of it, and a run at the default, are
    told nothing.
    """
    if plan is None or jobs <= 1:
        return
    for reason, modules in _serial_modules(plan):
        named = ", ".join(f"'{module}'" for module in modules)
        print(
            f"warning: --jobs {jobs} does not reach {named}: {_SERIAL_REASONS[reason]}",
            file=sys.stderr,
        )


def _console(args: argparse.Namespace) -> Console:
    """The narration this invocation speaks with: stderr, muted by -q/--quiet."""
    return Console(quiet=bool(getattr(args, "quiet", False)))


def _add_global_argument(
    subparser: argparse.ArgumentParser,
    text: str = "write the machine-wide lockfile instead of this project's",
) -> None:
    subparser.add_argument(
        "-g",
        "--global",
        action="store_true",
        dest="global_lock",
        help=text,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ffrwd", description=f"ffrwd {_version()} - SQL frontend for FFmpeg filtergraphs"
    )
    parser.add_argument("-V", "--version", action="version", version=f"ffrwd {_version()}")
    subparsers = parser.add_subparsers(dest="command")

    compile_p = subparsers.add_parser("compile", help="compile SQL to an ffmpeg command")
    _add_query_arguments(compile_p)
    _add_quiet_argument(compile_p)
    compile_p.add_argument(
        "--graph-only", action="store_true", help="print only the filter_complex string"
    )
    _add_jobs_argument(compile_p)
    explain_p = subparsers.add_parser("explain", help="dump the compiled IR graph as JSON")
    _add_query_arguments(explain_p)
    _add_quiet_argument(explain_p)
    explain_view = explain_p.add_mutually_exclusive_group()
    explain_view.add_argument(
        "--mermaid", action="store_true", help="print the graph as a mermaid flowchart"
    )
    explain_view.add_argument(
        "--diagram",
        action="store_true",
        help="render the flowchart in the terminal (needs the diagram extra)",
    )
    validate_p = subparsers.add_parser("validate", help="check that a query compiles")
    _add_query_arguments(validate_p)
    _add_quiet_argument(validate_p)
    validate_p.add_argument(
        "--json", action="store_true", dest="as_json", help="emit the error as JSON"
    )
    # `ffrwd -h` lands on run's help via the default dispatch, so run's
    # description carries the version the way the top-level one does.
    run_p = subparsers.add_parser(
        "run",
        help="compile and execute ffmpeg",
        description=f"ffrwd {_version()} - compile and execute ffmpeg (the default subcommand)",
    )
    _add_query_arguments(run_p)
    _add_quiet_argument(run_p)
    # No default: what an unset --timeout means is decided per compile, from
    # the inputs' durations (`_timeout`).
    run_p.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=f"ffmpeg timeout in seconds (default: {DEFAULT_TIMEOUT}s, or ten "
        "times the longest input, and none at all for a live input)",
    )
    run_p.add_argument(
        "-y", action="store_true", dest="overwrite", help="pass -y (overwrite) to ffmpeg"
    )
    _add_jobs_argument(run_p)
    showing = run_p.add_mutually_exclusive_group()
    showing.add_argument(
        "--show",
        action="store_true",
        help="write the files and play each video output in an ffplay window",
    )
    showing.add_argument(
        "--show-only",
        action="store_true",
        dest="show_only",
        help="play each video output and write nothing to disk",
    )
    # Not in the --show group: the conflict is refused as a typed error with
    # a hint (`remote.submit_run`), not as an argparse usage line.
    run_p.add_argument(
        "--remote",
        action="store_true",
        help="submit the run to the hosted runner instead of executing ffmpeg here",
    )

    jobs_p = subparsers.add_parser(
        "jobs", help="list, watch, cancel or fetch your remote runs"
    )
    jobs_mode = jobs_p.add_mutually_exclusive_group()
    jobs_mode.add_argument(
        "--json", action="store_true", dest="as_json", help="emit the listing as JSON"
    )
    jobs_mode.add_argument(
        "--watch",
        action="store_true",
        help="redraw the listing every few seconds until nothing is running",
    )
    jobs_mode.add_argument(
        "--cancel",
        metavar="ID",
        default=None,
        help="ask for a job's cancellation (a unique id prefix works)",
    )
    jobs_mode.add_argument(
        "--fetch",
        metavar="ID",
        default=None,
        help="download a succeeded job's outputs (a unique id prefix works)",
    )
    jobs_p.add_argument(
        "-y",
        action="store_true",
        dest="overwrite",
        help="overwrite existing files when fetching",
    )
    _add_quiet_argument(jobs_p)

    list_p = subparsers.add_parser(
        "list", help="print what this project and its dependencies provide"
    )
    list_p.add_argument(
        "--json", action="store_true", dest="as_json", help="emit the listing as JSON"
    )
    init_p = subparsers.add_parser(
        "init", help="write ffrwd.json, ffrwd.lock and a starter recipe here"
    )
    init_p.add_argument(
        "--name",
        default=None,
        help="the package name, <namespace>/<package> or just the package "
        "segment (default: this directory's name)",
    )
    init_p.add_argument(
        "--namespace",
        default=None,
        help="the namespace half of the name (default: derived from the git remote's owner)",
    )
    init_p.add_argument(
        "--rust",
        action="store_true",
        help="scaffold a wasm module package: the Rust crate that builds it, "
        "the SQL declaring it, and a recipe calling it",
    )

    search_p = subparsers.add_parser("search", help="find packages in the registry")
    search_p.add_argument(
        "term",
        nargs="?",
        default=None,
        help="matched against each package's name, description and "
        "function names (default: everything published)",
    )
    search_p.add_argument(
        "--json", action="store_true", dest="as_json", help="emit the results as JSON"
    )

    install_p = subparsers.add_parser("install", help="install a package from the registry")
    install_p.add_argument(
        "package",
        nargs="?",
        default=None,
        help="<namespace>/<package>, or <namespace>/<package>@<version> for an exact "
        "version; omitted, installs the project standing here -- its manifest's "
        "dependencies and the models it pins",
    )
    _add_global_argument(install_p)
    _add_quiet_argument(install_p)

    path_p = subparsers.add_parser(
        "path", help="print where an installed package's content is on this machine"
    )
    path_p.add_argument("package", help="<namespace>/<package>")
    _add_global_argument(path_p, "read the machine-wide lockfile instead of this project's")

    link_p = subparsers.add_parser("link", help="read a package live out of a directory")
    link_p.add_argument("path", help="the directory holding the package's ffrwd.json")
    _add_global_argument(link_p)
    unlink_p = subparsers.add_parser("unlink", help="drop a linked package")
    unlink_p.add_argument(
        "name", help="the linked package's name (or its directory) to stop reading live"
    )
    _add_global_argument(unlink_p)
    login_p = subparsers.add_parser("login", help="save the token this machine publishes with")
    login_p.add_argument(
        "--token",
        required=True,
        help="an ffrwd token minted in the dashboard",
    )
    subparsers.add_parser("logout", help="remove the saved token")
    publish_p = subparsers.add_parser(
        "publish", help="publish the package here to the registry"
    )
    _add_quiet_argument(publish_p)

    setup_p = subparsers.add_parser(
        "setup", help="download what a query needs before one asks for it"
    )
    setup_p.add_argument(
        "what",
        choices=["nn"],
        help="nn: the ONNX Runtime the modules that run models load",
    )
    setup_p.add_argument(
        "--cuda",
        action="store_true",
        help="also take the CUDA execution provider, which needs a CUDA 12 "
        "runtime and cuDNN 9 already on the machine",
    )
    setup_p.add_argument(
        "--full",
        action="store_true",
        help="with --cuda, also take a pinned CUDA 12 and cuDNN 9, for a "
        "machine that has neither",
    )
    _add_quiet_argument(setup_p)

    subparsers.add_parser("prompt", help="print the LLM system prompt for this dialect")
    mcp_p = subparsers.add_parser("mcp", help="serve the compiler to an editor or agent over MCP")
    mcp_p.add_argument(
        "--allow-unsafe",
        action="store_true",
        dest="allow_unsafe",
        help="also expose the tools that do more than answer about a query",
    )
    subparsers.add_parser(
        loudnorm.ENV_SUBCOMMAND,
        help="read ffmpeg's stderr on stdin, print loudnorm's measurements as exports",
    )

    return parser


def _read_file(path: str) -> str | None:
    """Read query text from `path` (or stdin for "-"). None + printed error on failure."""
    if path == "-":
        return sys.stdin.read()
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as err:
        print(f"error: could not read {path!r}: {err.strerror or err}", file=sys.stderr)
        return None


_VAR_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _parse_set_vars(pairs: list[str], command: str) -> tuple[dict[str, str] | None, int]:
    """Parse repeated ``-v/--set NAME=VALUE`` pairs into a dict (last wins).

    Returns ``(variables, 0)`` on success, or ``(None, 2)`` with a usage
    error already printed to stderr for a malformed pair: no ``=``, or a
    name outside ``[A-Za-z_][A-Za-z0-9_]*``.
    """
    variables: dict[str, str] = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep or _VAR_NAME_RE.fullmatch(name) is None:
            print(
                f"error: {command}: malformed -v/--set {pair!r}, want "
                "NAME=VALUE with NAME matching [A-Za-z_][A-Za-z0-9_]*",
                file=sys.stderr,
            )
            return None, 2
        variables[name] = value
    return variables, 0


def _project_start(args: argparse.Namespace) -> Path:
    """Where the upward walk for ``ffrwd.json`` starts.

    A ``-f PATH`` query is read from a file, and the project it belongs to is
    the one above that file's own directory. A positional query and ``-f -``
    were typed at the shell, so the walk starts at the working directory.
    """
    if args.file is not None and args.file != "-":
        return Path(args.file).parent
    return Path.cwd()


_LEADING_WORD_RE = re.compile(r"[A-Za-z]+")


def _starts_a_statement(text: str) -> bool:
    """True when `text` begins a SQL statement, past leading whitespace and comments.

    The whole of rule one: four words, and a positional starting with any of
    them is SQL whatever else it might have named. The scan skips what the
    lexer would skip so a query written under a ``--`` header still reads as
    one.
    """
    at = 0
    end = len(text)
    while at < end:
        if text[at].isspace():
            at += 1
        elif text.startswith("--", at):
            newline = text.find("\n", at)
            at = end if newline == -1 else newline + 1
        elif text.startswith("/*", at):
            close = text.find("*/", at + 2)
            at = end if close == -1 else close + 2
        else:
            break
    word = _LEADING_WORD_RE.match(text, at)
    return word is not None and word.group().lower() in STATEMENT_KEYWORDS


def _qualified_recipe(package: Package, recipe: str) -> str:
    """`recipe` written the way `run` reaches it: two segments for the default, three otherwise."""
    if recipe == package.package:
        return f"{package.namespace}.{package.package}"
    return f"{package.namespace}.{package.package}.{recipe}"


def _recipe_names(packages: PackageSet | None) -> list[str]:
    """Every recipe the discovered packages ship, each qualified."""
    if packages is None:
        return []
    return [
        _qualified_recipe(package, recipe)
        for name in packages.names()
        for package in [_package(packages, name)]
        for recipe in package.recipes
    ]


def _package(packages: PackageSet, name: str) -> Package:
    found = packages.get(name)
    assert found is not None  # a name `names()` just handed back
    return found


def _matching_recipes(name: str, packages: PackageSet | None) -> list[tuple[Package, str]]:
    """The (package, recipe name) pairs `name` names, qualified or bare.

    Three segments (``ns.pkg.recipe``) name one entry of a map `bin`; two
    (``ns.pkg``) name a string `bin`'s recipe. A bare name is looked up
    across every installed package, which may match more than one.
    """
    if packages is None:
        return []
    parts = name.split(".")
    if len(parts) in (2, 3):
        namespace, package_name = parts[0], parts[1]
        package = packages.find(namespace, package_name)
        if package is None:
            return []
        member = parts[2] if len(parts) == 3 else None
        recipe = package.recipe(member)
        return [] if recipe is None else [(package, package.package if member is None else member)]
    found: list[tuple[Package, str]] = []
    for claimed in packages.names():
        package = _package(packages, claimed)
        if package.recipe(name) is not None:
            found.append((package, name))
    return found


def _recipe_text(
    name: str, packages: PackageSet | None
) -> tuple[str, tuple[str, str]] | None:
    """The query text of the recipe `name` names, or None when it names none.

    Two packages shipping one name is a rejection rather than a pick: the
    qualified form says which, and guessing would run the wrong recipe.
    """
    found = _matching_recipes(name, packages)
    if not found:
        return None
    if len(found) > 1:
        written = ", ".join(_qualified_recipe(package, recipe) for package, recipe in found)
        raise FfrwdError(
            ErrorCode.UNSUPPORTED_SQL,
            f"more than one package ships a recipe named '{name}'",
            hint=f"name the one you mean: {written}",
        )
    package, recipe = found[0]
    file = package.recipe(recipe)
    assert file is not None  # _matching_recipes only keeps shipped recipes
    try:
        return file.read_text(encoding="utf-8"), (package.name, package.version)
    except OSError as err:
        raise FfrwdError(
            ErrorCode.UNSUPPORTED_SQL,
            f"recipe '{_qualified_recipe(package, recipe)}' could not be read: "
            f"{err.strerror or err}",
            hint=f"its file is {file}",
        ) from err


@dataclass(frozen=True)
class _Query:
    """What `_resolve_query` hands the subcommand handlers.

    `text` is the substituted query; `unset` is the substitution's map from an
    unset variable's NULL to its name, threaded into every compile so a
    rejection can say which variable was not set. `recipe` is the name the
    positional resolved to (None for inline SQL or ``-f``), and `source` the
    pre-substitution text, whose ``-- variables:`` header the error hint reads.
    """

    text: str
    unset: dict[tuple[int, int], str]
    recipe: str | None
    source: str
    #: The (name, version) of the package a recipe belongs to, so a call it
    #: makes reaches the version THAT package declares rather than whatever
    #: the project happens to have. None for inline SQL and for `-f`.
    owner: tuple[str, str] | None = None
    #: The ``-v name=value`` pairs as the user supplied them. `text` already
    #: has them substituted; a remote submit sends them raw as well, so the
    #: job's record holds both halves.
    variables: dict[str, str] = field(default_factory=dict)


def _recipe_variables_error(err: FfrwdError, name: str, text: str) -> FfrwdError:
    """`err` again, its hint naming what the recipe's own header declares."""
    declared = declared_variables(text)
    if not declared:
        return err
    written = ", ".join(variable.name for variable in declared)
    return FfrwdError(
        err.code,
        err.message,
        line=err.line,
        col=err.col,
        hint=f"'{name}' declares {written}; define each with -v name=value",
    )


def _resolve_query(
    args: argparse.Namespace,
) -> tuple[_Query | None, PackageSet | None, int]:
    """Resolve the query text and its project for compile/explain/validate/run.

    Exactly one of the positional ``query`` (inline SQL) or ``-f/--file`` is
    required. Returns ``(query, packages, 0)`` on success, or
    ``(None, None, exit_code)`` with the error already printed to stderr: 2
    for a usage violation (both or neither given; a malformed ``-v``; a ``-v``
    naming a variable the text never references), 1 for a file that could not
    be read.

    ``packages`` is the project the query was written in, or None when the walk
    finds no manifest -- the ordinary case, and the one where a compile is
    exactly what it was before projects existed.

    A positional that does not begin a statement and names a recipe one of
    those packages ships is that recipe: its file's text replaces it, and
    every subcommand taking a query takes a recipe name too.

    ``-v/--set`` substitution runs here, once, so every handler inherits it.
    An UNSET reference substitutes to NULL rather than failing -- absence,
    which the compile itself judges -- so the check points the other way: a
    ``-v`` for a name the text never references is the usage error, naming
    what the text does reference. A `FfrwdError` from a malformed manifest
    is not caught here; it propagates to the caller's own handling.
    """
    has_query = args.query is not None
    has_file = args.file is not None
    if has_query and has_file:
        print(
            f"error: {args.command}: give a SQL string or -f/--file, not both",
            file=sys.stderr,
        )
        return None, None, 2
    if not has_query and not has_file:
        print(
            f"error: {args.command}: give a SQL string or -f/--file",
            file=sys.stderr,
        )
        return None, None, 2
    if has_file:
        text = _read_file(args.file)
        if text is None:
            return None, None, 1
    else:
        assert args.query is not None
        text = args.query

    packages = discover(_project_start(args))
    recipe: str | None = None
    owner: tuple[str, str] | None = None
    if not has_file and not _starts_a_statement(text):
        shipped = _recipe_text(text, packages)
        if shipped is not None:
            recipe, (text, owner) = text, shipped

    variables, code = _parse_set_vars(args.set_vars, args.command)
    if variables is None:
        return None, None, code
    names = referenced(text)
    unknown = sorted(name for name in variables if name not in names)
    if unknown:
        what = f"recipe '{recipe}'" if recipe is not None else "the query"
        listed = (
            "it references " + ", ".join(f":{name}" for name in sorted(names))
            if names
            else "it references no variables"
        )
        written = ", ".join(f"{name}=..." for name in unknown)
        verb = "names a variable" if len(unknown) == 1 else "name variables"
        print(
            f"error: {args.command}: -v {written} {verb} {what} never "
            f"references; {listed}",
            file=sys.stderr,
        )
        return None, None, 2
    sub = substitute(text, variables)
    return (
        _Query(
            text=sub.text,
            unset=sub.unset,
            recipe=recipe,
            source=text,
            owner=owner,
            variables=variables,
        ),
        packages,
        0,
    )


def _maybe_print_file_hint(
    err: FfrwdError, source: str | None, packages: PackageSet | None = None
) -> None:
    """Suggest -f when an inline positional string names an existing file or
    ends in .sql/.SQL. Fires on ANY compile error, not just PARSE_ERROR: a
    bare filename like `query.sql` parses as a SQL column reference and fails
    as UNSUPPORTED_SQL. CLI sugar only -- never touches `err`.

    A positional that does not begin a statement and names no recipe is a
    recipe name that matched nothing, so the recipes there ARE to run are
    named too. One that DID match is not: the rejection came from inside that
    recipe, and a list of the others is noise over it."""
    if source is None:
        return
    if os.path.exists(source) or source.lower().endswith(".sql"):
        print(
            f"hint: '{source}' looks like a file; did you mean -f '{source}'?",
            file=sys.stderr,
        )
    if _starts_a_statement(source) or _matching_recipes(source, packages):
        return
    shipped = _recipe_names(packages)
    if shipped:
        print(f"hint: installed recipes: {', '.join(shipped)}", file=sys.stderr)


def _print_warnings(warnings: WarningLog) -> None:
    """Print what the compile had to say, to stderr.

    stderr and not stdout: `compile`'s stdout is the ffmpeg command, and
    `validate --json` is the repair loop's JSON. Printed after the command has
    run, so nothing interleaves with what it wrote.
    """
    for warning in warnings.warnings:
        print(f"warning: {warning.message}", file=sys.stderr)
        if warning.hint is not None:
            print(f"hint: {warning.hint}", file=sys.stderr)


def _print_error(
    err: FfrwdError,
    *,
    source: str | None = None,
    packages: PackageSet | None = None,
    query: _Query | None = None,
) -> None:
    # A recipe run by name gets the richer hint: an unset-variable rejection
    # names what the recipe's own `-- variables:` header declares.
    if query is not None and query.recipe is not None and unset_variable(err) is not None:
        err = _recipe_variables_error(err, query.recipe, query.source)
    print(f"error: {err}", file=sys.stderr)
    _maybe_print_file_hint(err, source, packages)


def _check_output_dir(out_path: str) -> str | None:
    """Return an error message if `out_path`'s parent directory does not exist.

    A destination containing "://" is a protocol URL (udp, rtmp, srt, ...):
    ffmpeg owns it, there is no directory to check.
    """
    if is_url(out_path):
        return None
    parent = Path(out_path).parent
    if str(parent) and not parent.exists():
        return f"error: output directory does not exist: {parent}"
    return None


def _sinks(graphs: list[Graph]) -> list[SinkUnit]:
    """Every command's sink units, in command order."""
    return [unit for graph in graphs for unit in graph.sinks]


def _needs_out_path(graphs: list[Graph]) -> bool:
    """True if some sink names no destination — i.e. the bare-SELECT case."""
    return any(unit.path is None for unit in _sinks(graphs))


def _output_paths(graphs: list[Graph]) -> list[str]:
    """Every file this command will write, for the directory-existence check."""
    return [unit.path for unit in _sinks(graphs) if unit.path is not None]


def _is_table_capable_query(
    text: str, packages: PackageSet | None, on_warning: OnWarning
) -> bool:
    """True if `text` succeeds as a table/csv query -- the fallback `compile`
    and `validate` try before giving up on a `compile_sql` error."""
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


_TABLE_USAGE_HINT = (
    "error: compile has nothing to show: this query has no media destination "
    "(no COPY, or every COPY is FORMAT csv); run it instead -- `ffrwd run ...` "
    "prints its result set as a table"
)
_NO_OUTPUT_PATH_ERROR = "error: no output path given: use COPY ... TO in the query"


def _print_table_sinks(sinks: list[TableSink]) -> int:
    """Print (or write) every sink of a table/csv query. `run`'s table half."""
    for sink in sinks:
        if not sink.csv:
            print(render_table(sink.result))
            continue
        text = render_csv(sink.result, header=sink.header)
        if sink.path is None:
            print(text, end="")  # already newline-terminated per row
            continue
        dir_error = _check_output_dir(sink.path)
        if dir_error is not None:
            print(dir_error, file=sys.stderr)
            return 1
        Path(sink.path).write_text(text, encoding="utf-8")
    return 0


def _cmd_compile(args: argparse.Namespace, on_warning: OnWarning) -> int:
    console = _console(args)
    query: _Query | None = None
    packages: PackageSet | None = None
    code = _check_jobs(args)
    if code != 0:
        return code
    try:
        query, packages, code = _resolve_query(args)
        if query is None:
            return code
        with console.status("compiling"):
            compiled = compile_all(
                query.text,
                packages=packages,
                on_warning=on_warning,
                unset=query.unset,
                owner=query.owner,
            )
            graphs = compiled.graphs
            emitted = emitted_commands(graphs)
    except FfrwdError as err:
        # A query with no streaming representation at all (metadata
        # columns, an un-COALESCEd join gap) fails HERE, so table mode is the
        # fallback -- tried only after compilation failed, and only for a
        # query that could BE one. If the fallback fails too, the original
        # error surfaces; it is usually more informative.
        # `query` is None only when `_resolve_query` raised before it could
        # return (a malformed manifest, an unreadable recipe), which cannot
        # be table-capable either, so it is guarded out of `classify`.
        if query is not None and _is_table_capable_query(query.text, packages, on_warning):
            print(_TABLE_USAGE_HINT, file=sys.stderr)
            return 2
        _print_error(err, source=args.query, packages=packages, query=query)
        return 1

    if args.graph_only:
        # One line per command; a compile is a sequence only for two_pass,
        # loudnorm2 and the copy-and-trim fan-out.
        print("\n".join(e.filter_complex for e in emitted))
        return 0

    # A bare SELECT compiles fine here (the streaming lowerer allows it), but
    # it has no COPY ... TO destination -- compile never invents one, so it
    # is the same refusal as the except branch above.
    if _needs_out_path(graphs):
        print(_TABLE_USAGE_HINT, file=sys.stderr)
        return 2
    # A query that reaches a wasm module is several processes, not one
    # command; `render_plan` prints them as the shell pipeline they are.
    if compiled.plan is not None:
        try:
            rendered = render_plan(
                compiled.plan,
                sidecar_argv=functools.partial(wasm.shown_argv, jobs=args.jobs),
            )
        except FfrwdError as err:
            _print_error(err, source=args.query, packages=packages, query=query)
            return 1
        print(rendered)
        _print_jobs_notice(compiled.plan, args.jobs)
        return 0
    print(_CHAIN.join(_shell_commands(emitted)))
    return 0


def _shell_commands(emitted: list[Emitted]) -> list[str]:
    """Every command of the compile as a shell-ready line, in order.

    ``shlex.join`` for all but a ``loudnorm2`` compile: there the measuring
    pass is wrapped in the ``eval "$(...)"`` that exports what it measured,
    and the write pass keeps its ``${FFRWD_LN_*}`` references expandable
    (:func:`ffrwd.loudnorm.shell_join`).
    """
    lines: list[str] = []
    for e in emitted:
        commands = build_ffmpeg_commands(e)
        if not e.measure_filter_complex:
            lines += [shlex.join(command) for command in commands]
            continue
        measure, *rest = commands
        lines.append(loudnorm.measure_command(shlex.join(measure)))
        lines += [loudnorm.shell_join(command) for command in rest]
    return lines


def _cmd_explain(args: argparse.Namespace, on_warning: OnWarning) -> int:
    console = _console(args)
    query: _Query | None = None
    packages: PackageSet | None = None
    try:
        query, packages, code = _resolve_query(args)
        if query is None:
            return code
        with console.status("compiling"):
            compiled = compile_all(
                query.text,
                packages=packages,
                on_warning=on_warning,
                unset=query.unset,
                owner=query.owner,
            )
    except FfrwdError as err:
        _print_error(err, source=args.query, packages=packages, query=query)
        return 1

    graphs = compiled.graphs
    if args.mermaid or args.diagram:
        text = diagram.render_diagram(graphs, compiled.plan)
        if args.mermaid:
            print(text)
            return 0
        if not diagram.termaid_available():
            print(f"error: {diagram.INSTALL_HINT}", file=sys.stderr)
            return 1
        print(diagram.render_terminal(text))
        return 0
    # One object for a single command, a JSON ARRAY for a sequence's. A query
    # that partitions into processes carries the plan beside its graph.
    payload: object = graphs[0].to_dict() if len(graphs) == 1 else [
        graph.to_dict() for graph in graphs
    ]
    if compiled.plan is not None:
        payload = {"graph": payload, "plan": compiled.plan.to_dict()}
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_validate(args: argparse.Namespace, on_warning: OnWarning) -> int:
    console = _console(args)
    query: _Query | None = None
    packages: PackageSet | None = None
    try:
        query, packages, code = _resolve_query(args)
        if query is None:
            return code
        with console.status("compiling"):
            compile_commands(
                query.text,
                packages=packages,
                on_warning=on_warning,
                unset=query.unset,
                owner=query.owner,
            )
    except FfrwdError as err:
        # "compiles = valid" still holds: a table/csv query compiles through
        # its own lenient pipeline, tried here exactly as in `_cmd_compile`.
        if query is not None and _is_table_capable_query(query.text, packages, on_warning):
            return 0
        if args.as_json:
            # Machine contract: stdout is pure JSON, the library error
            # verbatim. The file hint goes to stderr so it cannot perturb it.
            print(json.dumps(err.to_dict()))
            _maybe_print_file_hint(err, args.query, packages)
        else:
            _print_error(err, source=args.query, packages=packages, query=query)
        return 1

    return 0


def _nothing_to_show() -> FfrwdError:
    return FfrwdError(
        ErrorCode.NOTHING_TO_SHOW,
        "nothing to show: the query writes no video output file",
        line=1,
        col=1,
        hint="a window plays a COPY that writes video; drop the flag to run "
        "the query as it stands",
    )


def _player() -> str:
    """The ffplay a window is opened with. Raises when there is none."""
    ffplay = binaries.ffplay_path()
    if ffplay is None:
        raise FfrwdError(
            ErrorCode.PLAYER_NOT_FOUND,
            "ffplay not found",
            line=1,
            col=1,
            hint=binaries.FFPLAY_HINT,
        )
    return ffplay


def _with_windows(
    graphs: list[Graph], *, only: bool
) -> tuple[list[Graph], list[list[str] | None]]:
    """`graphs` with a display output each, and the ffplay reading it.

    The returned player list is parallel to the commands
    :func:`emitted_commands` will render, so a graph left with no sink -- one
    that writes only rows, or one `only` suppressed -- takes no entry.

    Raises when there is nothing to play, or nothing to play it with.
    """
    titles = [show.shown_path(g) for g in graphs]
    if not any(title is not None for title in titles):
        raise _nothing_to_show()

    ffplay = _player()
    shown = [insert_splits(show.with_display(g, only=only)) for g in graphs]
    players = [
        show.ffplay_argv(ffplay, title) if title is not None else None
        for graph, title in zip(shown, titles)
        if graph.sinks
    ]
    return shown, players


def _with_plan_windows(
    graphs: list[Graph], plan: ProcessPlan, *, only: bool
) -> tuple[list[Graph], ProcessPlan, dict[str, list[str]]]:
    """`plan` with a display output per shown process, and the ffplay reading each.

    The players are keyed by process id, which is how :func:`execute_plan`
    takes them. `graphs` come back suppressed under `only`, since they are
    what the destination checks read and nothing is written.

    Raises when there is nothing to play, or nothing to play it with.
    """
    titles = show.shown_processes(plan)
    if not titles:
        raise _nothing_to_show()

    ffplay = _player()
    return (
        [show.suppressed(g) for g in graphs] if only else graphs,
        show.with_plan_display(plan, only=only),
        {pid: show.ffplay_argv(ffplay, title) for pid, title in titles.items()},
    )


def _cmd_run(args: argparse.Namespace, on_warning: OnWarning) -> int:
    console = _console(args)
    query: _Query | None = None
    packages: PackageSet | None = None
    code = _check_jobs(args)
    if code != 0:
        return code
    try:
        query, packages, code = _resolve_query(args)
        if query is None:
            return code
        with console.status("compiling"):
            is_table_capable, _has_copy = classify(
                query.text,
                packages=packages,
                on_warning=on_warning,
                unset=query.unset,
                owner=query.owner,
            )
    except FfrwdError as err:
        _print_error(err, source=args.query, packages=packages, query=query)
        return 1

    # The remote fork: everything after this point -- probing, ffmpeg, output
    # directories -- is the runner's machine's business, not this one's.
    if args.remote:
        try:
            if is_table_capable:
                raise FfrwdError(
                    ErrorCode.UNSUPPORTED_SQL,
                    "a table query needs no cloud -- run it locally",
                    hint="it runs without ffmpeg: drop --remote and the result "
                    "set prints right here",
                )
            with console.status("submitting"):
                submitted = remote.submit_run(query, packages, args, announce=console.say)
        except FfrwdError as err:
            _print_error(err, source=args.query, packages=packages, query=query)
            return 1
        print(f"submitted {submitted.job_id}")
        if submitted.remaining is not None:
            print(remote.free_footer(submitted.remaining))
        print(f"follow: ffrwd jobs --watch    fetch: ffrwd jobs --fetch {submitted.job_id[:8]}")
        return 0

    showing = args.show or args.show_only

    # No media COPY -- a bare SELECT, or every COPY is FORMAT csv -- IS a
    # table query, always: the table/csv path, which needs no ffmpeg.
    if is_table_capable:
        if showing:
            _print_error(_nothing_to_show(), source=args.query, packages=packages, query=query)
            return 1
        try:
            sinks = compile_table_sql(
                query.text,
            packages=packages,
            on_warning=on_warning,
            unset=query.unset,
            owner=query.owner,
            )
        except FfrwdError as err:
            _print_error(err, source=args.query, packages=packages, query=query)
            return 1
        return _print_table_sinks(sinks)

    try:
        with console.status("compiling"):
            compiled = compile_all(
                query.text,
                packages=packages,
                on_warning=on_warning,
                unset=query.unset,
                owner=query.owner,
            )
        graphs: list[Graph] = compiled.graphs
        plan: ProcessPlan | None = compiled.plan
        players: list[list[str] | None] = []
        windows: dict[str, list[str]] = {}
        if showing and plan is not None:
            graphs, plan, windows = _with_plan_windows(graphs, plan, only=args.show_only)
        elif showing:
            graphs, players = _with_windows(graphs, only=args.show_only)
        emitted: list[Emitted] = emitted_commands(graphs)
    except FfrwdError as err:
        _print_error(err, source=args.query, packages=packages, query=query)
        return 1

    # A media COPY names its own destination, so this fires only for the rare
    # script that mixes a media COPY with a `COPY ... TO STDOUT WITH (FORMAT
    # csv)` sink -- STDOUT has no file path for a media run.
    if _needs_out_path(graphs):
        print(_NO_OUTPUT_PATH_ERROR, file=sys.stderr)
        return 2

    for path in _output_paths(graphs):
        dir_error = _check_output_dir(path)
        if dir_error is not None:
            print(dir_error, file=sys.stderr)
            return 1

    if binaries.ffmpeg_path() is None:
        print(f"error: ffmpeg not found: {binaries.INSTALL_HINT}", file=sys.stderr)
        return 1

    budget = _timeout(args, compiled)

    if plan is not None:
        return _run_plan(plan, args, packages, query, windows, budget, console)

    # `ffrwd.execute` owns the loop; the CLI owns the printing. stderr stays
    # uncaptured (`capture_stderr` left false) so ffmpeg writes its progress
    # straight to the terminal, and `echo` puts each `$ <cmd>` line in front
    # of the output it produced.
    result = execute(
        emitted,
        timeout=budget,
        overwrite=args.overwrite,
        echo=_echo_command,
        players=players,
        show_only=args.show_only,
    )

    if result.timed_out:
        print(f"error: ffmpeg timed out after {budget}s", file=sys.stderr)
        return 1
    if result.measure_error is not None:
        print(f"error: {result.measure_error}", file=sys.stderr)
        return 1
    if result.exit_code != 0:
        print(result.commands[-1].stderr, file=sys.stderr, end="")
        print(f"error: ffmpeg exited with code {result.exit_code}", file=sys.stderr)
        return result.exit_code

    return 0


def _timeout(args: argparse.Namespace, compiled: Compiled) -> float | None:
    """How long this run may take: ``--timeout``, or the compile's own budget.

    An explicit ``--timeout`` always wins. Unset, the budget is the compile's
    (:attr:`ffrwd.compiler.Compiled.default_timeout`), which is None -- no
    timeout -- for a run whose inputs have no duration to scale by.
    """
    if args.timeout is not None:
        timeout: float = args.timeout
        return timeout
    return compiled.default_timeout


def _provision_nn(plan: ProcessPlan, console: Console) -> int:
    """Put an ONNX Runtime on this machine when `plan` reaches a model.

    Before anything is spawned, so a first run of a query that infers waits
    for a download once instead of failing on a library it has not got. A
    plan that binds no model touches none of this and asks the sidecar
    nothing.
    """
    if not any(
        isinstance(process, SidecarProcess) and process.models
        for process in plan.processes
    ):
        return 0
    try:
        nn.ensure(announce=console.say)
    except FfrwdError as err:
        _print_error(err)
        return 1
    return 0


def _run_plan(
    plan: ProcessPlan,
    args: argparse.Namespace,
    packages: PackageSet | None,
    query: _Query | None,
    players: dict[str, list[str]],
    timeout: float | None,
    console: Console,
) -> int:
    """Run a query that reaches a wasm module, and report it as `run` reports.

    A plan's members share one terminal, so unlike `execute` their stderr is
    captured and only a FAILING stage's is printed -- several ffmpegs and a
    sidecar interleaving progress lines is nothing anyone can read. Losing one
    member closes the pipes around it, so every member that failed on its own
    is named, not just the first seen.

    `players` is the ffplay each shown process's stdout feeds, empty for a run
    that asked for no window. `timeout` is per stage, None for a run nothing
    bounds. ``--jobs`` reaches each sidecar through the renderer, and the
    modules it could not reach are named before anything is spawned.
    """
    code = _provision_nn(plan, console)
    if code != 0:
        return code
    _print_jobs_notice(plan, args.jobs)
    result = execute_plan(
        plan,
        sidecar_argv=functools.partial(wasm.sidecar_argv, jobs=args.jobs),
        timeout=timeout,
        overwrite=args.overwrite,
        echo=_echo_member,
        players=players,
        show_only=args.show_only,
    )
    if result.overflow is not None:
        print(f"error: {result.overflow}", file=sys.stderr)
        return 1
    if result.timed_out:
        print(f"error: the pipeline timed out after {timeout}s", file=sys.stderr)
        return 1
    if result.exit_code != 0:
        for member in result.failures:
            print(member.stderr_tail, file=sys.stderr)
            print(
                f"error: {member.id} exited with code {member.exit_code}: "
                f"{member.summary}",
                file=sys.stderr,
            )
        return result.exit_code
    return 0


def _echo_member(name: str, argv: list[str]) -> None:
    print(f"$ {name}:", shlex.join(argv))


def _echo_command(argv: list[str]) -> None:
    print("$", shlex.join(argv))


@dataclass(frozen=True)
class _ListedRecipe:
    """One recipe: the variables its query declares, split required from optional.

    The split is derived, not read off the header -- see :func:`_required_variables`.
    """

    name: str
    path: Path
    required: tuple[Variable, ...]
    optional: tuple[Variable, ...]


@dataclass(frozen=True)
class _Listed:
    """What one package provides, read once and printed either way."""

    package: Package
    functions: tuple[Signature, ...]
    recipes: tuple[_ListedRecipe, ...]


def _relative(path: Path, root: Path) -> str:
    """`path` as the manifest writes it, or its own text when it is elsewhere."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _listed(package: Package, packages: PackageSet | None) -> _Listed:
    """Read one package's exports and recipes. Raises like any other read.

    The export list is the manifest's; `package_signatures` parses the files
    only for the parameter types, and checks each export is defined where the
    manifest says. `packages` is the whole discovered set (not just `package`
    itself), the same one a real compile of one of its recipes would resolve
    namespaced calls against.
    """
    recipes = []
    for name, path in package.recipes.items():
        text = path.read_text(encoding="utf-8")
        declared = declared_variables(text)
        required = _required_variables(
            text, frozenset(variable.name for variable in declared), packages
        )
        recipes.append(
            _ListedRecipe(
                name=name,
                path=path,
                required=tuple(v for v in declared if v.name in required),
                optional=tuple(v for v in declared if v.name not in required),
            )
        )
    return _Listed(
        package=package, functions=package_signatures(package), recipes=tuple(recipes)
    )


def _listing_json(listed: list[_Listed]) -> str:
    """The listing as one JSON object, for scripting."""
    packages = [
        {
            "name": entry.package.name,
            "version": entry.package.version,
            "layer": entry.package.layer,
            "linked": entry.package.linked,
            "root": str(entry.package.root),
            "exports": [
                {
                    "name": signature.name,
                    "params": [
                        {
                            "name": param.name,
                            "type": param.type,
                            "default": param.written_default,
                        }
                        for param in signature.params
                    ],
                    "returns": signature.returns,
                    "file": _relative(signature.export, entry.package.root),
                }
                for signature in entry.functions
            ],
            "recipes": [
                {
                    "name": listed_recipe.name,
                    "file": _relative(listed_recipe.path, entry.package.root),
                    "required": [
                        {"name": variable.name, "description": variable.description}
                        for variable in listed_recipe.required
                    ],
                    "optional": [
                        {"name": variable.name, "description": variable.description}
                        for variable in listed_recipe.optional
                    ],
                }
                for listed_recipe in entry.recipes
            ],
            "dependencies": [
                {"name": name, "range": range_}
                for name, range_ in entry.package.dependencies.items()
            ],
        }
        for entry in listed
    ]
    return json.dumps({"packages": packages}, indent=2)


def _package_rows(listed: list[_Listed]) -> TableResult:
    rows: list[list[CellValue]] = [
        [
            entry.package.name,
            entry.package.version,
            entry.package.layer,
            entry.package.linked,
        ]
        for entry in listed
    ]
    return TableResult(columns=["package", "version", "layer", "linked"], rows=rows)


def _export_rows(listed: list[_Listed]) -> TableResult:
    rows: list[list[CellValue]] = [
        [
            entry.package.name,
            signature.written,
            signature.returns,
            _relative(signature.export, entry.package.root),
        ]
        for entry in listed
        for signature in entry.functions
    ]
    return TableResult(columns=["package", "export", "returns", "file"], rows=rows)


def _recipe_rows(listed: list[_Listed]) -> TableResult:
    rows: list[list[CellValue]] = [
        [
            entry.package.name,
            listed_recipe.name,
            _written_variables(listed_recipe.required),
            _written_variables(listed_recipe.optional),
            _relative(listed_recipe.path, entry.package.root),
        ]
        for entry in listed
        for listed_recipe in entry.recipes
    ]
    return TableResult(
        columns=["package", "recipe", "required", "optional", "file"], rows=rows
    )


def _dependency_rows(listed: list[_Listed]) -> TableResult:
    rows: list[list[CellValue]] = [
        [entry.package.name, name, range_]
        for entry in listed
        for name, range_ in entry.package.dependencies.items()
    ]
    return TableResult(columns=["package", "dependency", "range"], rows=rows)


def _written_variables(variables: tuple[Variable, ...]) -> str:
    """The declared variables as the header writes them, descriptions and all."""
    return ", ".join(
        f"{variable.name} ({variable.description})" if variable.description else variable.name
        for variable in variables
    )


def _cmd_list(args: argparse.Namespace, on_warning: OnWarning) -> int:
    """Print what the project at the working directory provides. Takes no query.

    The same upward walk every other subcommand does, so the answer is the one
    a compile here would resolve against.
    """
    try:
        found = discover(Path.cwd())
        packages = [] if found is None else [found.packages[name] for name in found.names()]
        listed = [_listed(package, found) for package in packages]
    except FfrwdError as err:
        _print_error(err)
        return 1
    except OSError as err:
        named = err.filename or "a file this project names"
        print(f"error: could not read {named}: {err.strerror or err}", file=sys.stderr)
        return 1

    if args.as_json:
        print(_listing_json(listed))
        return 0

    # One section per kind, each headed by its own name: four tables in a row
    # are unreadable without one.
    sections = [
        f"{heading}\n{render_table(table)}"
        for heading, table in (
            ("packages", _package_rows(listed)),
            ("exports", _export_rows(listed)),
            ("recipes", _recipe_rows(listed)),
            ("dependencies", _dependency_rows(listed)),
        )
    ]
    print("\n\n".join(sections))
    return 0


# The starter `init` writes: a recipe, not an export. A lib must name a file
# that defines its export, so a fresh directory has nothing to declare one
# with -- and a runnable recipe is the first thing there is to try.
_STARTER_RECIPE = "resize"
_STARTER_FILE = "recipes/resize.sql"
_STARTER_QUERY = """\
-- Scale a file's video to 720p, its audio carried through untouched.
-- variables: source (input media path), dest (output path)
-- example: ffrwd run resize -v source=in.mp4 -v dest=out.mp4
COPY (
  SELECT scale(f.video[1], -2, 720), f.audio[1]
  FROM input(:'source') f
) TO :'dest'
"""

# The wasm module scaffold `init --rust` writes on top of the manifest and the
# lockfile. One export and one recipe, both named for what the module does.
_RUST_EXPORT = "invert"
_RUST_EXPORT_FILE = f"src/{_RUST_EXPORT}.sql"
_RUST_RECIPE = "invert"
_RUST_RECIPE_FILE = f"recipes/{_RUST_RECIPE}.sql"
_RUST_BUILD_FILE = "build.rs"
_RUST_CARGO_FILE = "Cargo.toml"
_RUST_SOURCE_FILE = "src/lib.rs"
_RUST_IGNORE_FILE = store.IGNORE_NAME
_RUST_GITIGNORE_FILE = store.GITIGNORE_NAME

# The wit-bindgen the in-repo modules build with; the scaffold pins the same one.
_WIT_BINDGEN_VERSION = "0.57.1"

# Where a built module lands, and what the lib SQL therefore names.
_RUST_ARTIFACT = "target/wasm32-wasip2/release/{crate}.wasm"

_RUST_CARGO = f"""\
[package]
name = "{{crate}}"
version = "0.1.0"
edition = "2021"

# A wasm component: one cdylib, no binary.
[lib]
crate-type = ["cdylib"]

[dependencies]
wit-bindgen = "{_WIT_BINDGEN_VERSION}"

[profile.release]
opt-level = 3
lto = true
strip = true
"""

_RUST_BUILD = '''\
// Puts the `ffrwd:av` wit where `wit_bindgen::generate!` reads it, from
// whichever of the two sources is available: FFRWD_WIT_DIR when the
// environment names one, else the `ffrwd/wasm` package installed here.
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const WIT_DIR_ENV: &str = "FFRWD_WIT_DIR";
const WIT_PACKAGE: &str = "ffrwd/wasm";
const WIT_FILE: &str = "av.wit";

fn main() {
    println!("cargo::rerun-if-env-changed={WIT_DIR_ENV}");
    let source = match env::var_os(WIT_DIR_ENV) {
        Some(named) => PathBuf::from(named),
        None => installed_wit_dir(),
    }
    .join(WIT_FILE);
    println!("cargo::rerun-if-changed={}", source.display());

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let wit = manifest.join("wit");
    fs::create_dir_all(&wit).expect("create wit/");
    fs::copy(&source, wit.join(WIT_FILE))
        .unwrap_or_else(|err| panic!("copy {}: {err}", source.display()));
}

/// The `wit` directory of the installed `ffrwd/wasm` package, asked of ffrwd.
fn installed_wit_dir() -> PathBuf {
    let asked = Command::new("ffrwd")
        .args(["path", WIT_PACKAGE])
        .output()
        .unwrap_or_else(|err| {
            panic!("`ffrwd path {WIT_PACKAGE}` could not be run ({err}); set {WIT_DIR_ENV} instead")
        });
    if !asked.status.success() {
        panic!(
            "`ffrwd path {WIT_PACKAGE}` failed: {}",
            String::from_utf8_lossy(&asked.stderr).trim()
        );
    }
    let printed = String::from_utf8(asked.stdout).expect("a path, in utf-8");
    PathBuf::from(printed.trim()).join("wit")
}
'''

_RUST_SOURCE = '''\
wit_bindgen::generate!({
    path: "wit",
    world: "video-module",
});

use exports::ffrwd::av::filter::{FrameInfo, Guest, Meta, Outcome, Output, StreamInfo};

struct Invert;

// JSON Schema for the `params` string a call passes; this module takes none.
const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;

fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("invert takes no params, got: {other}")),
    }
}

impl Guest for Invert {
    // What the module is, read before anything runs. `pixel_formats` empty
    // would make this an audio module, and the two are never both.
    fn describe() -> Meta {
        Meta {
            name: "invert".to_string(),
            version: "0.1.0".to_string(),
            params_schema: PARAMS_SCHEMA.to_string(),
            rows_schema: String::new(),
            pixel_formats: vec!["rgba".to_string()],
            sample_formats: vec![],
            sample_rates: vec![],
            channel_counts: vec![],
            rows_language: vec![],
        }
    }

    // Once per instance, before any frame. The frame size and pixel format
    // are fixed from here on, so state sized to them is built here.
    fn init(
        _width: u32,
        _height: u32,
        _pix_fmt: String,
        _stream_info: StreamInfo,
        params: String,
    ) -> Result<(), String> {
        validate_params(&params)
    }

    // New parameters between frames; rejecting them leaves the old ones in force.
    fn set_params(params: String) -> Result<(), String> {
        validate_params(&params)
    }

    // True lets the host run frames in parallel, so it must be a promise.
    fn frame_independent() -> bool {
        true
    }

    // One frame in, one frame out. `Output::Passthrough` would hand the input
    // back uncopied; this one rewrites the bytes it was given.
    fn process(_info: FrameInfo, frame: Vec<u8>) -> Outcome {
        let mut out = frame;
        let (pixels, _) = out.as_chunks_mut::<4>();
        for pixel in pixels {
            pixel[0] = 255 - pixel[0];
            pixel[1] = 255 - pixel[1];
            pixel[2] = 255 - pixel[2];
        }
        Outcome {
            output: Output::Frame(out),
            rows: vec![],
        }
    }
}

export!(Invert);
'''

_RUST_EXPORT_SQL = """\
-- The wasm module this package ships, declared as a function a query calls.
-- The path is the cargo build output, relative to this package's root.
CREATE FUNCTION {export}(v video_stream) RETURNS video_stream
  AS '{artifact}', '{export}' LANGUAGE wasm;
"""

_RUST_RECIPE_QUERY = """\
-- Invert a file's picture, its audio carried through untouched.
-- variables: source (input media path), dest (output path)
-- example: ffrwd run {recipe} -v source=in.mp4 -v dest=out.mp4
COPY (
  SELECT {call}(f.video[1]), f.audio[1]
  FROM input(:'source') f
) TO :'dest'
"""

_RUST_IGNORE = """\
# Build output. The one file the lib SQL names is pulled back in by the
# manifest's closure; the rest of the directory stays out of the archive.
target/
"""

# `wit/` is written by build.rs, not by hand.
_RUST_GITIGNORE = """\
target/
wit/
"""

_RUST_README = """\
# {name}

TODO: what this package does, and how a query calls it.
"""


def _rust_scaffold(name: str) -> dict[str, str]:
    """The module scaffold's files, keyed by their path under the project."""
    segment = name.partition("/")[2]
    return {
        _RUST_CARGO_FILE: _RUST_CARGO.format(crate=segment),
        _RUST_BUILD_FILE: _RUST_BUILD,
        _RUST_SOURCE_FILE: _RUST_SOURCE,
        _RUST_EXPORT_FILE: _RUST_EXPORT_SQL.format(
            export=_RUST_EXPORT, artifact=_RUST_ARTIFACT.format(crate=segment)
        ),
        _RUST_RECIPE_FILE: _RUST_RECIPE_QUERY.format(
            recipe=_RUST_RECIPE, call=f"{name.replace('/', '.')}.{_RUST_EXPORT}"
        ),
        _RUST_IGNORE_FILE: _RUST_IGNORE,
        _RUST_GITIGNORE_FILE: _RUST_GITIGNORE,
        README_NAME: _RUST_README.format(name=name),
    }


# The scaffold's paths, which do not depend on the name: what `init --rust`
# refuses to overwrite, checked before a name is worked out.
_RUST_PATHS = (
    _RUST_CARGO_FILE,
    _RUST_BUILD_FILE,
    _RUST_SOURCE_FILE,
    _RUST_EXPORT_FILE,
    _RUST_RECIPE_FILE,
    _RUST_IGNORE_FILE,
    _RUST_GITIGNORE_FILE,
    README_NAME,
)

_NOT_A_NAMESPACE_HINT = (
    "a namespace is a lowercase plain identifier: a letter or underscore, then "
    "letters, digits or underscores -- pass --namespace to give one"
)


def _folded_identifier(name: str) -> str:
    """`name` folded to a plain identifier: lowercase, everything else an underscore."""
    return re.sub(r"[^a-z0-9_]", "_", name.lower())


def _git_remote_owner(directory: Path) -> str | None:
    """The owner segment of the git remote `origin`'s URL, or None.

    One cheap subprocess; anything that goes wrong -- no git, no repository,
    no remote, an unparseable URL -- is None, never an error: it only feeds a
    default the flag overrides.
    """
    git = shutil.which("git")
    if git is None:
        return None
    try:
        result = subprocess.run(
            [git, "-C", str(directory), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    url = result.stdout.strip()
    if url.endswith(".git"):
        url = url[: -len(".git")]
    if "://" in url:
        url = url.split("://", 1)[1]
    # scp-like `git@host:owner/repo` becomes `git@host/owner/repo`.
    parts = [part for part in url.replace(":", "/").split("/") if part]
    return parts[-2] if len(parts) >= 2 else None


def _checked_namespace(written: str, given: str) -> str | None:
    """`written` as a usable namespace, or None with the rejection printed."""
    usable = is_namespace(written) and any(char.isalnum() for char in written)
    if not usable:
        print(f"error: init: {given} gives no namespace: {written!r}", file=sys.stderr)
        print(f"hint: {_NOT_A_NAMESPACE_HINT}", file=sys.stderr)
        return None
    if written in RESERVED_NAMESPACES:
        reserved = ", ".join(sorted(RESERVED_NAMESPACES))
        print(f"error: init: namespace {written!r} is reserved", file=sys.stderr)
        print(
            f"hint: {reserved} belong to ffrwd itself; pass --namespace with another",
            file=sys.stderr,
        )
        return None
    return written


def _init_name(args: argparse.Namespace, directory: Path) -> tuple[str | None, str]:
    """The ``<namespace>/<package>`` name `init` writes, and where its namespace came from.

    ``--name`` with a slash is the whole name; otherwise the package segment
    is ``--name``'s or the directory's, and the namespace is ``--namespace``'s
    or the git remote's owner. None with the rejection already printed.
    """
    written_name = str(args.name) if args.name is not None else None
    if written_name is not None and "/" in written_name:
        if args.namespace is not None:
            print(
                "error: init: --name carries the namespace; give one or the other",
                file=sys.stderr,
            )
            return None, ""
        namespace, _, segment = written_name.partition("/")
        checked = _checked_namespace(namespace, "--name")
        if checked is None:
            return None, ""
        return _init_full_name(checked, segment, "--name"), "--name"

    segment_source = written_name if written_name is not None else directory.name
    if not segment_source:
        print(f"error: init: {directory} has no name to take the package's from", file=sys.stderr)
        print("hint: pass --name", file=sys.stderr)
        return None, ""
    segment = _folded_identifier(segment_source)
    if not (is_namespace(segment) and any(char.isalnum() for char in segment)):
        print(
            f"error: init: {segment_source!r} gives no package segment: {segment!r}",
            file=sys.stderr,
        )
        print("hint: pass --name with a package name", file=sys.stderr)
        return None, ""

    if args.namespace is not None:
        chosen = _checked_namespace(str(args.namespace), "--namespace")
        if chosen is None:
            return None, ""
        return _init_full_name(chosen, segment, "--namespace"), "--namespace"

    owner = _git_remote_owner(directory)
    derived = _checked_silently(_folded_identifier(owner)) if owner is not None else None
    if derived is None:
        print("error: init: no namespace to name the package under", file=sys.stderr)
        print(
            "hint: pass --namespace, or --name <namespace>/<package>; none could be "
            "derived from a git remote here",
            file=sys.stderr,
        )
        return None, ""
    return _init_full_name(derived, segment, "the git remote"), "the git remote 'origin'"


def _checked_silently(written: str) -> str | None:
    """`written` as a usable namespace, or None -- for a derived default only."""
    usable = is_namespace(written) and any(char.isalnum() for char in written)
    return written if usable and written not in RESERVED_NAMESPACES else None


def _init_full_name(namespace: str, segment: str, given: str) -> str | None:
    """The two halves joined and checked, or None with the rejection printed."""
    if not (is_namespace(segment) and any(char.isalnum() for char in segment)):
        print(f"error: init: {given} gives no package segment: {segment!r}", file=sys.stderr)
        print(
            "hint: each half of a package name is a lowercase plain identifier",
            file=sys.stderr,
        )
        return None
    name = f"{namespace}/{segment}"
    refused = name_refusal(name)
    if refused is not None:
        print(f"error: init: {refused[0]}", file=sys.stderr)
        print(f"hint: {refused[1]}", file=sys.stderr)
        return None
    return name


def _cmd_init(args: argparse.Namespace, on_warning: OnWarning) -> int:
    """Write the files a project starts as, into the working directory.

    The manifest, an empty lockfile and a starter recipe; ``--rust`` writes a
    wasm module package instead of the bare one -- the crate that builds the
    module, the SQL declaring it, and a recipe calling it.
    """
    directory = Path.cwd()
    rust = bool(args.rust)
    relative = _RUST_PATHS if rust else (_STARTER_FILE,)
    written = [
        directory / MANIFEST_NAME,
        directory / LOCKFILE_NAME,
        *(directory / one for one in relative),
    ]
    for path in written:
        if path.exists():
            print(f"error: init: {path} already exists", file=sys.stderr)
            print(
                "hint: init writes a project from scratch and overwrites nothing",
                file=sys.stderr,
            )
            return 1

    name, namespace_source = _init_name(args, directory)
    if name is None:
        return 1

    files = _rust_scaffold(name) if rust else {_STARTER_FILE: _STARTER_QUERY}
    manifest, lockfile = written[0], written[1]
    writing = manifest
    try:
        for one, text in files.items():
            writing = directory / one
            writing.parent.mkdir(parents=True, exist_ok=True)
            writing.write_text(text, encoding="utf-8", newline="\n")
        writing = manifest
        if rust:
            write_manifest(
                manifest,
                name=name,
                version="0.1.0",
                lib={_RUST_EXPORT: _RUST_EXPORT_FILE},
                bin={_RUST_RECIPE: _RUST_RECIPE_FILE},
                dependencies={wasm.WIT_PACKAGE: wasm.WORLD_VERSION},
                keywords=[],
                capabilities=[],
            )
        else:
            write_manifest(
                manifest,
                name=name,
                version="0.1.0",
                bin={_STARTER_RECIPE: _STARTER_FILE},
            )
        writing = lockfile
        write_lockfile(lockfile, ())
        # What was just written has to read back, or the next command refuses
        # a project this one made.
        read_manifest(manifest)
        read_lockfile(lockfile)
    except FfrwdError as err:
        _print_error(err)
        return 1
    except OSError as err:
        print(
            f"error: init: {writing} could not be written: {err.strerror or err}",
            file=sys.stderr,
        )
        return 1

    if rust:
        print(f"wrote {MANIFEST_NAME}, {LOCKFILE_NAME} and the module scaffold in {directory}")
    else:
        print(f"wrote {MANIFEST_NAME}, {LOCKFILE_NAME} and {_STARTER_FILE} in {directory}")
    print(
        f"package '{name}' (namespace from {namespace_source}); a project that installs "
        f"it calls its functions as {name.replace('/', '.')}.name()"
    )
    if rust:
        print(f"take the wit the module builds against: ffrwd install {wasm.WIT_PACKAGE}")
        print("build the module: cargo build --target wasm32-wasip2 --release")
        print(
            f"then run the recipe: ffrwd run {_RUST_RECIPE} "
            f"-v source=in.mp4 -v dest=out.mp4"
        )
        return 0
    print(
        f"run the starter recipe: ffrwd run {_STARTER_RECIPE} "
        f"-v source=in.mp4 -v dest=out.mp4"
    )
    return 0


def _lock_to_write(args: argparse.Namespace) -> tuple[Path | None, int]:
    """The lockfile `link`/`unlink` writes, or None with the usage error printed.

    Never creates one as a side effect: outside a project there is nothing to
    record a package in, and inventing a lockfile in whatever directory they
    happen to stand in is not a project.
    """
    if args.global_lock:
        return store.global_lock_path(), 0
    found = find_lockfile(Path.cwd())
    if found is None:
        print(
            f"error: {args.command}: no {LOCKFILE_NAME} in {Path.cwd()} or above it",
            file=sys.stderr,
        )
        print(
            f"hint: `ffrwd {args.command} -g ...` works machine-wide, usable from any "
            f"directory; `ffrwd init` starts a project here that pins its own",
            file=sys.stderr,
        )
        return None, 2
    return found, 0


def _held_entries(path: Path) -> tuple[LockEntry, ...]:
    """What `path` already pins, or nothing when there is no file there yet."""
    return read_lockfile(path).entries if path.is_file() else ()


def _held_dependencies(path: Path) -> dict[str, str]:
    """What `path`'s own project directly installed, carried over by a rewrite that isn't one."""
    return dict(read_lockfile(path).dependencies) if path.is_file() else {}


def _described(entry: LockEntry) -> str:
    """What an entry being replaced was, for the line that says it is going."""
    if isinstance(entry, LinkEntry):
        return f"the link to {entry.path}"
    return f"the installed {entry.name} {entry.version}"


def _search_rows(listings: tuple[packages_module.Listing, ...]) -> TableResult:
    rows: list[list[CellValue]] = [
        [listing.name, listing.version, listing.installs_week, listing.description]
        for listing in listings
    ]
    return TableResult(
        columns=["package", "version", "installs/week", "description"], rows=rows
    )


def _cmd_search(args: argparse.Namespace, on_warning: OnWarning) -> int:
    """Print what the registry ranks for `term`. Reads nothing local."""
    try:
        found = packages_module.search(args.term)
    except FfrwdError as err:
        _print_error(err)
        return 1

    if args.as_json:
        payload = {
            "registry": packages_module.base_url(),
            "packages": [listing.to_dict() for listing in found],
        }
        print(json.dumps(payload, indent=2))
        return 0
    # An empty table, not a rejection: a term nothing matches is an answer.
    print(render_table(_search_rows(found)))
    return 0


def _install_here(args: argparse.Namespace) -> int:
    """Bare ``ffrwd install``: make the project standing here whole.

    Fetches the dependencies its manifest pins, at their written versions,
    the models it pins beside its own modules, and the runtime those modules
    load -- everything installing this package from the registry would have
    fetched, so a fresh clone builds and publishes after this.
    """
    console = _console(args)
    if args.global_lock:
        print("error: install: -g needs a package name", file=sys.stderr)
        print(
            "hint: bare `ffrwd install` installs the project standing here; a "
            "machine-wide install names what to fetch",
            file=sys.stderr,
        )
        return 2
    manifest = find_manifest(Path.cwd())
    if manifest is None:
        print(
            f"error: install: no {MANIFEST_NAME} in {Path.cwd()} or above it",
            file=sys.stderr,
        )
        print(
            "hint: name a package to install one from the registry, or run "
            "`ffrwd init` to start a project here",
            file=sys.stderr,
        )
        return 2
    lock = manifest.parent / LOCKFILE_NAME
    try:
        with console.status("installing"):
            installed = packages_module.install_project(
                manifest, lock=lock, announce=console.say
            )
    except FfrwdError as err:
        _print_error(err)
        return 1

    package = installed.package
    print(f"installed what {package.name} {package.version} needs in {lock}")
    if installed.brought:
        brought = ", ".join(f"{one.name} {one.version}" for one in installed.brought)
        print(f"  fetched: {brought}")
    else:
        print("  all dependencies were already pinned")
    return 0


def _cmd_install(args: argparse.Namespace, on_warning: OnWarning) -> int:
    """Install a package into this project's lockfile, or -g's, and record it.

    Fetches what it depends on too, recursively -- each at its highest
    published version, unless the lockfile already pins that exact version.
    With no package named, installs the project standing here instead.
    """
    if args.package is None:
        return _install_here(args)
    console = _console(args)
    lock, code = _lock_to_write(args)
    if lock is None:
        return code
    manifest = lock.parent / MANIFEST_NAME
    try:
        with console.status("installing"):
            installed = packages_module.install(
                args.package,
                lock=lock,
                manifest=manifest if manifest.is_file() else None,
                announce=console.say,
            )
    except FfrwdError as err:
        _print_error(err)
        return 1

    release = installed.release
    print(f"installed {release.name} {release.version} in {lock}")
    if installed.replaced is not None:
        print(f"  replacing {_described(installed.replaced)}")
    if not installed.downloaded:
        print("  its content was already in the store; nothing was downloaded")
    if installed.manifest is not None:
        print(f"  recorded in {installed.manifest.name} as a dependency")
    if installed.brought:
        brought = ", ".join(f"{one.name} {one.version}" for one in installed.brought)
        print(f"  brought along as dependencies: {brought}")
    full = release.name.replace("/", ".")
    print(f"a query calls it as {full}.<name>() -- `ffrwd list` shows what it provides")
    return 0


def _lock_to_read(args: argparse.Namespace) -> tuple[Path | None, int]:
    """The lockfile `path` resolves through, or None with the usage error printed.

    Install's discovery, reading rather than writing: the project's own
    lockfile, or the machine-wide one with ``-g``.
    """
    if args.global_lock:
        return store.global_lock_path(), 0
    found = find_lockfile(Path.cwd())
    if found is None:
        print(
            f"error: {args.command}: no {LOCKFILE_NAME} in {Path.cwd()} or above it",
            file=sys.stderr,
        )
        print(
            f"hint: run `ffrwd {args.command} -g ...` for a package installed on this "
            f"machine, or `ffrwd install` here first",
            file=sys.stderr,
        )
        return None, 2
    return found, 0


def _cmd_path(args: argparse.Namespace, on_warning: OnWarning) -> int:
    """Print where an installed package's content is, one line and nothing else.

    A build script reads it -- `ffrwd path ffrwd/wasm` is how a module's
    ``build.rs`` finds the wit -- so the line is the path alone.
    """
    name = str(args.package)
    if not is_package_name(name):
        _print_error(
            FfrwdError(
                ErrorCode.UNSUPPORTED_SQL,
                f"path: {name!r} is not a package name",
                hint="a package is named <namespace>/<package>, each half a lowercase "
                "plain identifier",
            )
        )
        return 1
    lock, code = _lock_to_read(args)
    if lock is None:
        return code
    try:
        entry = held_entry(_held_entries(lock), name, lock)
        if entry is None:
            _print_error(
                FfrwdError(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"path: {lock} pins no package '{name}'",
                    hint=f"run `ffrwd install {name}` to put its content in the store",
                )
            )
            return 1
        root = entry_root(entry, lock)
    except FfrwdError as err:
        _print_error(err)
        return 1
    print(root)
    return 0


def _written_link_path(target: Path, lock: Path, *, relative: bool) -> str:
    """How the lockfile writes the linked directory.

    Relative to the lockfile for a project's own, which keeps the file
    readable beside the tree it points into; absolute for the machine-wide
    one, which sits under the cache directory where relative would only be a
    climb. A path on another drive has no relative form and stays absolute.
    """
    resolved = target.resolve()
    if not relative:
        return str(resolved)
    try:
        return Path(os.path.relpath(resolved, lock.parent)).as_posix()
    except ValueError:
        return str(resolved)


def _cmd_link(args: argparse.Namespace, on_warning: OnWarning) -> int:
    """Record a package read live out of `path`, in this project's lockfile or -g's."""
    target = Path(args.path)
    manifest = target / MANIFEST_NAME
    if not manifest.is_file():
        print(f"error: link: {target} holds no {MANIFEST_NAME}", file=sys.stderr)
        print("hint: link the directory a package's manifest sits in", file=sys.stderr)
        return 1
    lock, code = _lock_to_write(args)
    if lock is None:
        return code

    try:
        package = read_manifest(manifest)
        entries = _held_entries(lock)
        replaced = held_entry(entries, package.name, lock)
        entry = LinkEntry(
            path=_written_link_path(target, lock, relative=not args.global_lock),
        )
        write_lockfile(
            lock, with_entry(entries, entry, replaced), dependencies=_held_dependencies(lock)
        )
    except FfrwdError as err:
        _print_error(err)
        return 1

    print(f"linked {package.name} -> {entry.path} in {lock}")
    if replaced is not None:
        print(f"  replacing {_described(replaced)}")
    return 0


def _linked_as(entry: LinkEntry, lock: Path) -> str:
    """How one link is named to the user: the package's name, or its bare path."""
    name = stored_name(entry, lock)
    return name if name is not None else f"the unreadable link {entry.path!r}"


def _matching_link(entries: tuple[LockEntry, ...], written: str, lock: Path) -> LinkEntry | None:
    """The link entry `written` names -- by package name, or by directory."""
    for entry in entries:
        if not isinstance(entry, LinkEntry):
            continue
        if stored_name(entry, lock) == written or entry.path == written:
            return entry
        # A dead link is still removable by the directory it points at.
        try:
            if (lock.parent / Path(entry.path)).resolve() == Path(written).resolve():
                return entry
        except (OSError, ValueError):
            continue
    return None


def _cmd_unlink(args: argparse.Namespace, on_warning: OnWarning) -> int:
    """Drop the link `name` names and rewrite the lockfile."""
    lock, code = _lock_to_write(args)
    if lock is None:
        return code
    try:
        entries = _held_entries(lock)
    except FfrwdError as err:
        _print_error(err)
        return 1

    held = _matching_link(entries, args.name, lock)
    if held is None:
        installed = held_entry(entries, args.name, lock)
        why = "" if installed is None else " -- it is installed, not linked"
        print(f"error: unlink: nothing links '{args.name}' in {lock}{why}", file=sys.stderr)
        linked = [
            _linked_as(entry, lock) for entry in entries if isinstance(entry, LinkEntry)
        ]
        print(
            f"hint: linked: {', '.join(linked)}" if linked else "hint: nothing here is linked",
            file=sys.stderr,
        )
        return 1

    try:
        write_lockfile(lock, without_entry(entries, held), dependencies=_held_dependencies(lock))
    except FfrwdError as err:
        _print_error(err)
        return 1
    print(f"unlinked '{args.name}' from {lock}")
    return 0


def _cmd_login(args: argparse.Namespace, on_warning: OnWarning) -> int:
    """Save the token this machine publishes with. Reaches nothing.

    The token's shape is checked and nothing else: whether it is live, and
    what it may do, is the registry's answer at the first command that uses
    it.
    """
    token = str(args.token)
    if not credentials.is_token(token):
        print("error: login: that is not an ffrwd token", file=sys.stderr)
        print(f"hint: {credentials.TOKEN_HINT}", file=sys.stderr)
        return 2
    try:
        path = credentials.save(token, api=packages_module.api_url())
    except FfrwdError as err:
        _print_error(err)
        return 1
    print(f"saved a token for {packages_module.api_url()} in {path}")
    print("`ffrwd publish` uses it; `ffrwd logout` removes it")
    return 0


def _cmd_logout(args: argparse.Namespace, on_warning: OnWarning) -> int:
    """Remove the saved token. Saying there was none is an answer, not a failure."""
    try:
        removed = credentials.clear()
    except FfrwdError as err:
        _print_error(err)
        return 1
    path = credentials.credentials_path()
    print(f"removed the token in {path}" if removed else f"no token was saved in {path}")
    if os.environ.get(credentials.TOKEN_ENV):
        print(
            f"note: {credentials.TOKEN_ENV} is set in this environment and still "
            "answers for this machine",
            file=sys.stderr,
        )
    return 0


def _cmd_publish(args: argparse.Namespace, on_warning: OnWarning) -> int:
    """Validate the package here, pack it, and upload it in one request.

    Everything that can be checked locally is, before anything leaves: the
    manifest, the exports, the modules, every recipe compiled against a
    synthetic file, and every dependency resolved in the registry. What comes
    back from the registry is its own refusal or its own confirmation.
    """
    console = _console(args)
    manifest = find_manifest(Path.cwd())
    if manifest is None:
        print(f"error: publish: no {MANIFEST_NAME} in {Path.cwd()} or above it", file=sys.stderr)
        print(
            "hint: publish from the directory a package's manifest sits in; "
            "`ffrwd init` starts one",
            file=sys.stderr,
        )
        return 2
    try:
        with console.status("validating"):
            prepared = publish_module.prepare(
                manifest,
                discover(manifest.parent),
                on_warning=on_warning,
                announce=console.say,
            )
    except FfrwdError as err:
        _print_error(err)
        return 1
    except OSError as err:
        named = err.filename or "a file this package names"
        print(f"error: publish: could not read {named}: {err.strerror or err}", file=sys.stderr)
        return 1

    package = prepared.package
    print(f"validated {package.name} {package.version}: {prepared.size} bytes, {prepared.sha256}")
    try:
        with console.status("uploading"):
            published = publish_module.publish(prepared, announce=console.say)
    except FfrwdError as err:
        _print_error(err)
        return 1
    print(f"published {published.name} {published.version} ({published.visibility})")
    print(f"install it with `ffrwd install {published.name}`")
    return 0


def _cmd_setup(args: argparse.Namespace, on_warning: OnWarning) -> int:
    """Download what a query would otherwise fetch on its way to running.

    Nothing depends on this having been run -- a query that reaches a model
    provisions the same tiers itself. It is for a CI image, a machine about to
    lose its network, and the Windows CUDA tier, which is five times the size
    of the DirectML one and never fetched unasked.
    """
    console = _console(args)
    if args.full and not args.cuda:
        print(
            "error: --full adds the libraries the CUDA provider needs; pass --cuda too",
            file=sys.stderr,
        )
        return 2
    try:
        found = nn.info()
        tiers = list(nn.wanted_tiers(found))
        if args.cuda and "cuda" not in tiers:
            tiers.append("cuda")
        if args.full:
            tiers.append("full")
        with console.status("provisioning"):
            directory = nn.provision(tiers, announce=console.say, found=found)
    except FfrwdError as err:
        _print_error(err)
        return 1

    if args.cuda and "cuda" in nn.wanted_tiers(found):
        print("--cuda: this platform already takes the CUDA provider without asking")
    print(f"ONNX Runtime {found.ort_version} for {found.platform} is in {directory}")
    for line in _runtime_listing(directory):
        print(f"  {line}")
    print(f"a query that runs a model finds it there; {nn.TARGET_VAR} names the target")
    return 0


def _runtime_listing(directory: Path) -> list[str]:
    """Every library under the runtime directory, with its size, in path order."""
    found = sorted(p for p in directory.rglob("*") if p.is_file())
    return [
        f"{path.relative_to(directory).as_posix()}  {path.stat().st_size:,} bytes"
        for path in found
    ]


def _cmd_prompt(args: argparse.Namespace, on_warning: OnWarning) -> int:
    print(build_system_prompt(registry_module.load()))
    return 0


def _cmd_mcp(args: argparse.Namespace, on_warning: OnWarning) -> int:
    """Serve MCP over stdin/stdout; takes no query.

    stdout is the protocol stream from here on, so this handler prints
    nothing to it -- the missing-SDK message goes to stderr like every other
    CLI error, before the server would have started.
    """
    from . import mcp as mcp_module

    if not mcp_module.sdk_available():
        print(f"error: {mcp_module.INSTALL_HINT}", file=sys.stderr)
        return 1
    mcp_module.serve(allow_unsafe=args.allow_unsafe)
    return 0


def _cmd_jobs(args: argparse.Namespace, on_warning: OnWarning) -> int:
    """List, watch, cancel or fetch remote runs. `ffrwd.remote` owns the work."""
    console = _console(args)
    try:
        return remote.jobs_command(args, announce=console.say)
    except FfrwdError as err:
        _print_error(err)
        return 1


def _cmd_loudnorm2env(args: argparse.Namespace, on_warning: OnWarning) -> int:
    """stdin (ffmpeg's stderr) -> the ``export FFRWD_LN_*=`` block.

    The other half of the printed ``loudnorm2`` command line. Nothing else
    calls it: ``run`` parses the same text through the same function without
    a shell in between.
    """
    try:
        values = loudnorm.parse(sys.stdin.read())
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    print(loudnorm.export_lines(values))
    return 0


_HANDLERS = {
    "compile": _cmd_compile,
    "explain": _cmd_explain,
    "validate": _cmd_validate,
    "run": _cmd_run,
    "jobs": _cmd_jobs,
    "list": _cmd_list,
    "init": _cmd_init,
    "search": _cmd_search,
    "install": _cmd_install,
    "path": _cmd_path,
    "link": _cmd_link,
    "unlink": _cmd_unlink,
    "login": _cmd_login,
    "logout": _cmd_logout,
    "publish": _cmd_publish,
    "setup": _cmd_setup,
    "prompt": _cmd_prompt,
    "mcp": _cmd_mcp,
    loudnorm.ENV_SUBCOMMAND: _cmd_loudnorm2env,
}


if __name__ == "__main__":
    sys.exit(main())
