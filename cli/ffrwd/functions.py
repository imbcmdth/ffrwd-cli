"""User-defined SQL functions: ``CREATE FUNCTION``, and what a call to one becomes.

A function is a parameterized query fragment, which is what a view cannot be::

    CREATE FUNCTION normalize_lang(raw text) RETURNS text AS $$
      SELECT CASE WHEN raw = 'en' THEN 'eng' ELSE raw END
    $$ LANGUAGE sql;

There is no runtime concept and nothing new in the IR: every call site INLINES
the body with its arguments bound, the definitions are lifted out of the
script, and the ordinary compiler runs on what is left. :func:`expanded` is the
whole surface -- ``ffrwd.parser.resolve`` wraps its own work in it, and
nothing else in the package knows functions exist.

Three things make that inlining honest:

**Hygiene.** Names live in ONE flat script-wide namespace, so a body that binds
``g`` breaks the moment the function is called twice. Every alias the body
binds is renamed per call site (``g`` -> ``first_track_1_g``), against a set of
names already taken by the script, so two calls never collide -- and two calls
to a body that declares its own ``input()`` mint two ``-i`` entries, since
input identity is the alias. A body may not reference an alias of the calling
query at all; it sees its parameters and its own FROM items, and nothing else.

**The body's own type is what compiles.** Inlining means the expansion IS the
query the user could have written by hand: a body selecting a bare array splats
in the SELECT list because a bare array column does, not because ``RETURNS
audio_stream[]`` said so. ``RETURNS`` is checked against the declared type
vocabulary (:mod:`ffrwd.types`) and decides table-returning from
value-returning; it is not re-checked against the body.

**Diagnostics through two layers.** A body's nodes carry positions in the
BODY's coordinates, which mean nothing in a script the reader is looking at.
Each expansion re-stamps them into a private high line range that encodes which
expansion they came from and which body line they were on (:data:`_BODY_LINE_BASE`,
the same trick as a ``#line`` directive). A rejection landing on one is
rewritten to anchor on the CALL SITE, its message saying which body line it
came from; on a successful resolve the range is flattened to the call site
outright, so no later pass can report a line the reader cannot find. A
rejection whose blame includes a written ARGUMENT anchors on the argument
instead, since ``ffrwd.parser._pos`` takes the earliest position in the
subtree and the caller's own text always sorts first.

**A table-returning function is a row source, so it becomes a CTE.**
``RETURNS TABLE(col type, ...)`` is called in FROM and nowhere else. Its body
is not spliced into the calling query -- a body with an ``unnest`` produces
MANY rows, and splicing would hand the host the body's FROM items without the
body ever being a relation of its own. Each call becomes one generated CTE
holding the whole body, its projections aliased to the declared column names,
and the FROM item becomes a reference to that CTE under the writer's alias.
Everything about cardinality is then the CTE row source's, already defined:
comma is a cross join with real multiplicity, ``WHERE`` narrows the product,
``array_agg``/``GROUP BY`` gather it, and several rows into one path is
``ROW_COUNT_MISMATCH``. A call in the SELECT list is a typed rejection: SQL
reads ``(f(x)).a`` once per FIELD, and input identity here is the alias, so
each read would mint its own ``-i`` for one file.

**A definition need not be written in the script.** A qualified call that
resolves in a package (``me.pick(...)``, :mod:`ffrwd.project`) reaches a
definition read out of that package's lib files and inlines through this
same expander -- same hygiene, same source map, same arity and type checks.
Nothing is spliced into the script and the flat script namespace is untouched,
because a package's definitions never enter it. What a package EXPORTS is its
manifest's ``lib``, a string or a map: each exported name must be defined in
the file the manifest names for it -- checked here, where the file is parsed
-- and every other definition in a lib file is private to the package.

A ``LANGUAGE wasm`` definition in a lib file has no body to inline, so it is
ADOPTED rather than expanded (:meth:`_Expander._adopt`): the call is rewritten
to name a declaration the script now holds, under the call path it was written
as, and lowering resolves it exactly as it resolves one the script declared.
Its module path is the one thing that cannot be read the script's way -- a
package is compiled from whatever working directory the caller is in -- so it
is resolved against the DEFINING PACKAGE's root when the lib file is read, and
a path leaving that root is a rejection.

One rule differs by where a definition was written. A package's lib files are
a LIBRARY: it exports more than any one query calls, so an uncalled definition
in one is the point of it. A definition in the user's own script is a script's,
and one nothing calls stays an error. :meth:`_Expander._uncalled` is where that
asymmetry is spelled out.
"""

from __future__ import annotations

import copy
import difflib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import UnionType

from sqlglot import exp

from .errors import ErrorCode, FfrwdError
from .ir import StreamType
from .parser import (
    _ARITHMETIC,
    FILTER_NAMESPACE,
    MACRO_NAMESPACE,
    SINK_STREAMS,
    ModuleExport,
    _check_query_args,
    _error,
    _ident_name,
    _pos,
    _statements,
    from_entries,
    is_annotation_argument,
    parse,
)
from .project import (
    RESERVED_NAMESPACES,
    Package,
    PackageSet,
    leaves_package,
    under_package,
)
from .types import (
    CUE_TYPE,
    STREAM_ARRAY_COLUMNS,
    TAGS_COLUMN,
    TYPES,
    element_type,
    is_array,
)
from .warnings import FfrwdWarning, OnWarning, WarningCode

__all__ = [
    "NAMEABLE_TYPES",
    "SINK_STREAMS",
    "WASM_STREAM_NAMES",
    "WASM_STREAM_TYPES",
    "Annotation",
    "AnnotationField",
    "Parameter",
    "Script",
    "Signature",
    "WasmFunction",
    "expanded",
    "package_modules",
    "package_signatures",
]

# The FROM item that mints an `-i`. Never an argument: it is a table, and a
# table reference in a value position is not SQL.
_INPUT = "input"

# Names a definition may not claim: the dialect's own FROM item, the two
# reserved namespaces, and every name sqlglot parses as a builtin (a call to
# one comes back as its own node type, never as the anonymous call expansion
# looks for, so redefining it would silently do nothing).
_RESERVED = frozenset({_INPUT, FILTER_NAMESPACE, MACRO_NAMESPACE})

# The types a signature may name: the scalars, the four stream records, and
# the non-stream records the compiler surfaces. Handles, maps and `container`
# have no spelling a query can write; `attachment` and `cue` are declared but
# not wired up.
NAMEABLE_TYPES: tuple[str, ...] = tuple(
    sorted(
        name
        for name, declared in TYPES.items()
        if declared.kind in {"scalar", "stream", "record"}
        and (not declared.fields or any(f.exposed for f in declared.fields))
    )
)

_TYPE_HINT = (
    "a signature names " + ", ".join(NAMEABLE_TYPES) + ", or an array of one "
    "(e.g. audio_stream[])"
)
_FUNCTION_HINT = (
    "write CREATE FUNCTION <name>(<param> <type>, ...) RETURNS <type> "
    "AS $$ <query> $$ LANGUAGE sql"
)
_WASM_HINT = (
    "write CREATE FUNCTION <name>(<stream> video_stream, ...) RETURNS "
    "video_stream AS '<module>.wasm', '<export>' LANGUAGE wasm, or the same "
    "with audio_stream throughout"
)
# What a signature taking several streams may not also do. One axis at a time:
# the annotation column and the multi-stream signature are separate features.
_WASM_MULTI_HINT = (
    "a multi-input module takes no annotation column yet: declare its streams, "
    "then the values it is configured with"
)
# The languages a body may be written in, and the stream types a module filters.
_SQL = "sql"
_WASM = "wasm"
_WASM_STREAM = "video_stream"
_WASM_AUDIO_STREAM = "audio_stream"
# The return type of a module that is a COPY destination: streams go in and
# nothing comes out -- the module's own effects are the product.
WASM_SINK = "sink"
_WASM_SINK_HINT = (
    "a sink function is a COPY destination: COPY (SELECT <streams>) TO "
    "<name>(<values>)"
)
# The return type of a module that is a FROM-position row source: the mirror
# of a sink -- it produces streams and reads none.
WASM_SOURCE = "source"
_WASM_SOURCE_HINT = (
    "a source produces streams and reads none: its parameters are text, "
    "number or boolean, the values it is configured with"
)
_WASM_SOURCE_CALL_HINT = "a source is called in FROM: FROM <name>(<values>) <alias>"
# A ROWS function: an array of annotation records in, one out, no stream. The
# name a message gives its RETURNS, which has no column name of its own.
_ROWS_RETURN = "the rows it returns"
_WASM_ROWS_HINT = (
    "a rows function reads one array of annotation records and returns one: "
    "<name>(<column> cue[]) RETURNS cue[]"
)
_WASM_ROWS_CALL_HINT = (
    "call a rows function over the annotation column a module produces, e.g. "
    "<name>(<producer>(<stream>).<column>)"
)
# What a module filters, keyed by the type its signature names. A module takes
# one kind of stream and returns the same kind.
WASM_STREAM_TYPES: Mapping[str, StreamType] = {
    _WASM_STREAM: "video",
    _WASM_AUDIO_STREAM: "audio",
}
# The same map read the other way: the type a signature writes for one kind.
WASM_STREAM_NAMES: Mapping[StreamType, str] = {
    kind: written for written, kind in WASM_STREAM_TYPES.items()
}
_WASM_STREAM_HINT = " or ".join(WASM_STREAM_TYPES)
# What a table-returning wasm function is told; a value return is now wired
# (text, number or boolean), so only TABLE is left unguessed at.
_WASM_VALUE_HINT = (
    f"a wasm function returns one {_WASM_STREAM_HINT}, a STRUCT of that same "
    "stream and one array of annotation records, sink as a COPY destination, "
    "or text, number or boolean as a compile-time value"
)
# A value-returning wasm function's parameters: the same scalar domain an
# annotation field draws from, since both are values a JSON Schema can hold.
_WASM_VALUE_PARAM_HINT = (
    "a value-returning wasm function's parameters are text, number or boolean"
)
_WASM_VALUE_DEFAULT_HINT = (
    "a value-returning wasm function runs at every call site; there is no "
    "default to fall back to when the caller omits an argument"
)
# The types an annotation record's fields may be: values, never streams.
_ANNOTATION_FIELD_TYPES = ("boolean", "number", "text")
_ANNOTATION_FIELD_HINT = (
    "an annotation record's fields are values: " + ", ".join(_ANNOTATION_FIELD_TYPES)
)
_TABLE_HINT = (
    "write RETURNS TABLE(<column> <type>, ...), one name per column the body selects"
)
_VALUE_BODY_HINT = "a value-returning function's body is one SELECT of one column"
_TABLE_BODY_HINT = (
    "a table-returning function's body is one SELECT, one column per RETURNS TABLE column"
)
_ARG_HINT = (
    "a function that needs a file takes its path as text and calls input() in "
    "its own FROM"
)
_BODY_SCOPE_HINT = (
    "a function body sees its parameters and its own FROM items; pass what it "
    "needs as an argument"
)

# The sqlglot data types the three scalars land as. Everything else is an
# unknown type name -- including the other spellings of these (`varchar`,
# `int`), which the dialect does not have.
_SCALAR_TYPES = {
    exp.DataType.Type.TEXT: "text",
    exp.DataType.Type.DECIMAL: "number",
    exp.DataType.Type.BOOLEAN: "boolean",
}

_ArgumentShape = type[exp.Expr] | tuple[type[exp.Expr], ...] | UnionType

# What a written argument's SHAPE says its type is, where the shape says
# anything at all. A column, a subscript, an accessor or a CASE says nothing
# here -- their types come from the probe or from their branches, and resolve
# checks them after expansion.
_ARGUMENT_KINDS: list[tuple[_ArgumentShape, str]] = [
    (exp.Boolean, "boolean"),
    (exp.DPipe, "text"),
    (exp.Cast, "text"),
    (_ARITHMETIC, "number"),
    (exp.Neg, "number"),
    (exp.Upper, "text"),
    (exp.Lower, "text"),
    (exp.Length, "number"),
    (exp.Round, "number"),
    (exp.Replace, "text"),
    (exp.Substring, "text"),
    (
        (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Between, exp.In,
         exp.Is, exp.And, exp.Or, exp.Not),
        "boolean",
    ),
]

# How each kind is named back to the writer in a rejection.
_KIND_NAMES = {
    "text": "a string",
    "number": "a number",
    "boolean": "true or false",
    "stream": "a stream",
}

# Body positions live in this line range, one SPAN per expansion, so a
# rejection landing on body text can be traced back to the call it came from.
# Well above any hand-written script's line count; :meth:`_Expander.settle`
# clears the range before any later pass can see it.
_BODY_LINE_BASE = 1_000_000
_BODY_LINE_SPAN = 10_000

# How many inlinings one script may do. Recursion is caught by name, so this
# only bounds a legal but absurd nesting.
_EXPANSION_BUDGET = 200


@dataclass(frozen=True)
class AnnotationField:
    """One column of an annotation record: a name and a value type."""

    name: str
    type: str


def _cue_annotation_fields() -> tuple[AnnotationField, ...]:
    """What one row of ``cue[]`` carries: the cue fields a query may write.

    A cue's `index` is the document's own numbering, which a module's rows do
    not carry, so the read-only field stays out.
    """
    return tuple(
        AnnotationField(f.name, f.type) for f in TYPES[CUE_TYPE].fields if f.writable
    )


@dataclass(frozen=True)
class Annotation:
    """An array of annotation records: what it is called and what one row carries.

    Frame annotations are typed columns of the query, not plumbing: a module
    that reads rows off each frame declares them as a return field, and a
    module that consumes them declares them as a parameter. The two are
    matched by their FIELDS at compile time. The name is the writer's own and
    reaches nothing at run time -- annotations travel as NDJSON either way.
    """

    name: str
    fields: tuple[AnnotationField, ...]

    @property
    def written(self) -> str:
        """The type as a signature spells it, without the column name."""
        inner = ", ".join(f"{f.name} {f.type}" for f in self.fields)
        return f"STRUCT({inner})[]"


@dataclass(frozen=True)
class Parameter:
    """One position of a signature: a name, a declared type, and an optional default.

    `default` is the literal a ``DEFAULT`` constraint declared, or None with
    no such constraint -- only a parameter carries one; a ``RETURNS TABLE``
    column never does. `annotation` is the record shape an array-of-struct
    type declares, and None for every other type.
    """

    name: str
    type: str
    default: exp.Expr | None = None
    annotation: Annotation | None = None

    @property
    def written_default(self) -> str | None:
        """The DEFAULT literal as written, or None with no default."""
        return None if self.default is None else _written(self.default)


def _check_sink_streams(
    name: str,
    streams: tuple[Parameter, ...],
    identifier: exp.Identifier,
    create: exp.Create,
) -> None:
    """A sink's stream parameters: either kind, and an array of either.

    A sink is a filter with no output pads and reads streams the way one
    does, except that its count comes from the query rather than from its own
    declaration. An ARRAY parameter takes every stream of its kind the SELECT
    carries, so it is the last of that kind a signature can name -- anything
    after it would always be handed nothing.
    """
    seen: dict[str, Parameter] = {}
    for param in streams:
        kind = element_type(param.type)
        taken = seen.get(kind)
        if taken is not None and is_array(taken.type):
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"wasm function '{name}' takes '{param.name}' after "
                f"'{taken.name}', which is {taken.type}",
                identifier,
                fallback=create,
                hint=f"an array parameter takes every {kind} the SELECT "
                f"carries, so nothing of that kind can follow it",
            )
        seen[kind] = param


def _leading_streams(params: tuple[Parameter, ...]) -> tuple[Parameter, ...]:
    """The run of stream parameters a signature opens with.

    An annotation column is not one, however stream-shaped its type reads:
    it is a record beside the stream, and it ends the run.
    """
    end = 0
    for param in params:
        if param.annotation is not None or _declared_kind(param.type) != "stream":
            break
        end += 1
    return params[:end]


@dataclass(frozen=True)
class WasmFunction:
    """One ``LANGUAGE wasm`` declaration: a module, an export, and a signature.

    `module` is the path the sidecar is given. Declared in the script, that is
    the path exactly as written, read against the working directory the way
    ``input()``'s is; declared in a package's lib file, it is the written path
    resolved against the package's own root, since a package's modules ship
    inside it. For a STREAM function `params` is the whole signature: the
    LEADING run are the streams the module filters (:attr:`stream_params`),
    and the rest are the module's own parameters. A VALUE function
    (:attr:`is_value`) filters nothing, so every parameter is its own -- there
    is no leading stream to skip. `line` and `col` are the declaration's, so a
    rejection about what the module turned out to declare anchors where the
    query named it.
    """

    name: str
    module: str
    export: str
    params: tuple[Parameter, ...]
    returns: str
    line: int
    col: int
    # The annotation column the RETURNS declares beside the stream, and None
    # for a module that reads no rows off its frames.
    emits: Annotation | None = None
    # What the stream field of an annotating RETURNS was named. Unused at run
    # time, like the annotation's own name, and kept so the signature reads back.
    stream_field: str = ""
    # The record the RETURNS declares for a ROWS function -- one that reads
    # rows and writes rows, with no stream anywhere. None for every other kind.
    returns_rows: Annotation | None = None
    # Which statement of the script declared this, the same counter a sql
    # function's carries: what "used before it is defined" compares against.
    position: int = 0
    # The package whose lib file declared this, and that package's own version.
    # Both "" for a declaration the script wrote, whose module path is read
    # against the working directory.
    package: str = ""
    package_version: str = ""

    @property
    def is_value(self) -> bool:
        """True for a function returning a compile-time value, not a stream."""
        return (
            self.returns not in WASM_STREAM_TYPES
            and not self.is_sink
            and not self.is_source
            and not self.is_rows
        )

    @property
    def is_rows(self) -> bool:
        """True for a ROWS function: rows in, rows out, no stream anywhere.

        It is neither a filter nor a compile-time value: the module runs
        beside the one producing its rows, in that sidecar, and its result is
        a row column of the declared record.
        """
        return self.returns_rows is not None

    @property
    def rows_param(self) -> Annotation:
        """The row column a ROWS function reads. Asking of any other is a bug."""
        annotation = self.params[0].annotation if self.params else None
        if not self.is_rows or annotation is None:
            raise ValueError(f"'{self.name}' does not read rows")
        return annotation

    @property
    def is_sink(self) -> bool:
        """True for a ``RETURNS sink`` function: a COPY destination.

        A sink reads its streams like any consumer and writes nothing back --
        its own effects are the product -- so it is legal only after ``TO``.
        """
        return self.returns == WASM_SINK

    @property
    def is_source(self) -> bool:
        """True for a ``RETURNS source`` function: a FROM-position row source.

        The mirror of a sink: it produces streams and reads none, so it takes
        only value parameters and is legal only in FROM.
        """
        return self.returns == WASM_SOURCE

    @property
    def stream_kind(self) -> StreamType:
        """The kind of stream this function filters.

        A sink names no return stream, so its kind is its first parameter's.
        Only a STREAM function or a sink reading named stream parameters has
        one; a value function is folded at compile time and filters nothing,
        so asking is a caller's mistake. A sink reading both kinds has no
        single one, and callers that care read :attr:`stream_params` instead.
        A sink declaring no stream parameters reads its rows straight off the
        SELECT list, and a source's kinds come from its own catalog, not its
        signature -- neither has one to give here.
        """
        if self.is_source:
            raise ValueError(
                f"'{self.name}' returns source; its kinds come from its "
                "catalog, not its signature"
            )
        if self.is_sink and not self.stream_params:
            raise ValueError(
                f"'{self.name}' reads rows from the SELECT list; its kinds "
                "come from the rows, not its signature"
            )
        written = self.params[0].type if self.is_sink and self.params else self.returns
        kind = WASM_STREAM_TYPES.get(element_type(written))
        if kind is None:
            raise ValueError(f"'{self.name}' returns {self.returns}, not a stream")
        return kind

    @property
    def reads_rows_from_select(self) -> bool:
        """True for a sink whose SELECT list supplies the row cells directly.

        A ``RETURNS sink`` declaring no stream parameters names only the
        values it is configured with; the COPY's SELECT list is read as the
        rows themselves, one cell per column, rather than as named stream
        arguments.
        """
        return self.is_sink and not self.stream_params

    @property
    def stream_kinds(self) -> tuple[StreamType, ...]:
        """The kind each stream parameter reads, in declaration order."""
        return tuple(
            WASM_STREAM_TYPES[element_type(param.type)] for param in self.stream_params
        )

    @property
    def reads_many(self) -> bool:
        """Whether any stream parameter is an ARRAY, and so reads several.

        Only a sink may declare one: a filter's pad count is the module's own,
        and an array has no number until the query is lowered.
        """
        return any(is_array(param.type) for param in self.stream_params)

    @property
    def called(self) -> str:
        """The path a call writes, which is what a message about one names it by.

        ``ns.pkg.fn`` while the declaration is still the package's, and the
        bare name for one the script wrote. Adoption renames a package's
        declaration to this same path, so the two agree afterwards.
        """
        if not self.package:
            return self.name
        return f"{self.package.replace('/', '.')}.{self.name}"

    @property
    def stream_params(self) -> tuple[Parameter, ...]:
        """The streams the module reads at once, in call order.

        The leading run of the signature. One for an ordinary module, several
        for one the sidecar hands more than one stream at a time; a value
        function filters none.
        """
        if self.is_value or self.is_rows:
            return ()
        return _leading_streams(self.params)

    @property
    def stream_arity(self) -> int:
        """How many stream arguments a call writes before the module's values."""
        return len(self.stream_params)

    @property
    def reads(self) -> Annotation | None:
        """The annotation column this function takes BESIDE a stream.

        None for a ROWS function, whose rows are the whole argument rather
        than a column riding one: :attr:`rows_param` is that one.
        """
        if self.is_value or self.is_rows:
            return None
        after = self.stream_arity
        return self.params[after].annotation if len(self.params) > after else None

    @property
    def reads_optional(self) -> bool:
        """Whether the annotation column carries DEFAULT NULL.

        An optional column lets one declaration serve both call shapes: over
        a producer the rows are wired in, over a plain stream nothing is.
        """
        if self.is_value or self.is_rows:
            return False
        after = self.stream_arity
        if len(self.params) <= after:
            return False
        column = self.params[after]
        return column.annotation is not None and column.default is not None

    @property
    def value_params(self) -> tuple[Parameter, ...]:
        """The parameters that become the module's own.

        A value function has no stream to skip: every parameter is its own.
        A stream function's leading streams, and its annotation column where
        one is declared, are neither -- they are not a value the module is
        configured with. A rows function's one parameter is its rows, so it
        has none either.
        """
        if self.is_rows:
            return ()
        if self.is_value:
            return self.params
        skip = self.stream_arity + (1 if self.reads is not None else 0)
        return self.params[skip:]

    @property
    def written_params(self) -> tuple[Parameter, ...]:
        """The parameters a CALL writes an argument for.

        A value function's are all of them, and so are a rows function's. A
        stream function's annotation column is not one: the call producing it
        produces the stream beside it, so a single written argument covers both.
        """
        if self.is_value or self.is_rows:
            return self.params
        return (*self.stream_params, *self.value_params)

    @property
    def annotation_written_params(self) -> tuple[Parameter, ...]:
        """The parameters a call writes when it writes the annotation column too.

        :attr:`written_params` plus the column itself, back in its declared
        position. This is what a call spells when the rows it hands over are
        not simply the ones its stream argument arrived with -- a filtered
        gather, say, which no single argument can cover.
        """
        if self.is_value or self.reads is None:
            return self.written_params
        return (
            *self.stream_params,
            self.params[self.stream_arity],
            *self.value_params,
        )

    @property
    def written_returns(self) -> str:
        """The return type as a signature spells it, annotation column included."""
        if self.emits is None:
            return self.returns
        stream, annotation = self.stream_field, self.emits
        return f"STRUCT({stream} {self.returns}, {annotation.name} {annotation.written})"

    @property
    def signature(self) -> str:
        written = ", ".join(_written_param(p) for p in self.params)
        return f"{self.name}({written}) RETURNS {self.written_returns}"


@dataclass(frozen=True)
class Script:
    """What one expansion produced: the query, and what it declared.

    `wasm` is keyed by the name a call in the expanded query writes: the
    script's own declaration names, and the full ``ns.pkg.fn`` path of every
    package declaration a call adopted. A script with no ``LANGUAGE wasm``
    function and no call into a package's leaves it empty.
    """

    tree: exp.Expr
    wasm: dict[str, WasmFunction] = field(default_factory=dict)


@dataclass
class _Function:
    """One ``CREATE FUNCTION``, validated, with its body already parsed.

    `node` is the name identifier -- the earliest positioned token of the
    statement, so a definition-level rejection anchors on the name.
    `aliases` is what the body binds and expansion has to rename.
    `columns` is the ``RETURNS TABLE`` list, and None for a value.
    `package` is the name of the package this came from, and "" for the
    script's own; `package_version` is that package's own version, alongside
    it -- together they are the exact instance whose scope a bare name inside
    the body sees, since two versions of one name are never the same scope.
    """

    name: str
    params: tuple[Parameter, ...]
    returns: str
    body: exp.Select
    node: exp.Expr
    aliases: frozenset[str]
    position: int
    columns: tuple[Parameter, ...] | None = None
    used: bool = False
    package: str = ""
    package_version: str = ""

    @property
    def returns_rows(self) -> bool:
        return self.columns is not None

    @property
    def library(self) -> bool:
        """True for a definition read out of a package's lib file, not the script."""
        return bool(self.package)

    @property
    def identity(self) -> tuple[str, str] | None:
        """(name, version) of the owning package, or None for the script's own."""
        return (self.package, self.package_version) if self.package else None

    @property
    def namespace(self) -> str:
        """The first segment of the owning package's name, and "" for the script's."""
        return self.package.partition("/")[0]

    @property
    def qualified(self) -> str:
        """The full call path: ``ns.pkg.fn`` for a package's, ``fn`` for a script's.

        Always the three-segment form, which reaches any export regardless of
        whether the calling project bound an alias to it -- unlike an alias,
        it never collides between two packages sharing a namespace.
        """
        return f"{self.package.replace('/', '.')}.{self.name}" if self.package else self.name

    @property
    def signature(self) -> str:
        written = ", ".join(_written_param(p) for p in self.params)
        return f"{self.qualified}({written}) RETURNS {self.returns}"


@dataclass(frozen=True)
class _Expansion:
    """One inlining: which function, and where it was written."""

    name: str
    line: int
    col: int


@contextmanager
def expanded(
    tree: exp.Expression,
    *,
    packages: PackageSet | None = None,
    on_warning: OnWarning | None = None,
    owner: tuple[str, str] | None = None,
) -> Iterator[Script]:
    """Yield `tree` with every function definition lifted out and every call inlined.

    `packages` is where a namespaced call resolves; None means the caller named
    no project, and a qualified call is then whatever it always was.
    `on_warning` hears what a resolution cost -- a global package, a linked one
    -- once per package; None is silence.
    `owner` is the (name, version) of the package the script itself ships in
    -- a recipe compiled by name -- so its qualified calls resolve at the
    versions that package declares. The script's own definitions stay its own
    either way; None for inline SQL and ``-f``.

    A rejection raised while the block runs is re-anchored: one landing inside
    an expanded body comes back pointing at the call site, saying which body
    line it came from. On a clean exit the body line range is flattened to the
    call site, so nothing downstream can report a position the reader cannot
    find.

    A script with no ``CREATE FUNCTION`` and no packages to call into yields
    the tree untouched.

    A ``LANGUAGE wasm`` function has no body to inline: the declaration is
    lifted out like any other and comes back on :attr:`Script.wasm`, and its
    calls are left where they were written for lowering to resolve.
    """
    expander = _Expander(packages=packages, on_warning=on_warning, owner=owner)
    try:
        # Expansion's own rejections need translating too: a call written
        # inside a body was already stamped by the expansion around it.
        expanded_tree = expander.run(tree)
    except FfrwdError as err:
        raise expander.translate(err) from err
    script = Script(tree=expanded_tree, wasm=dict(expander.wasm))
    try:
        yield script
    except FfrwdError as err:
        raise expander.translate(err) from err
    expander.settle(expanded_tree)


# -- reading a definition -------------------------------------------------


def _create_kind(create: exp.Create) -> str:
    kind = create.args.get("kind")
    return kind.upper() if isinstance(kind, str) else ""


def _written(node: exp.Expr | None) -> str:
    """What a node says when printed back, for a message. Never raises."""
    if node is None:
        return "?"
    try:
        return node.sql(dialect="postgres")
    except Exception:  # a node sqlglot cannot render is still a rejection
        return node.__class__.__name__.upper()


def _written_param(param: Parameter) -> str:
    """One signature position as written: ``name type``, or with its DEFAULT."""
    if param.written_default is None:
        return f"{param.name} {param.type}"
    return f"{param.name} {param.type} DEFAULT {param.written_default}"


def _listed(columns: tuple[Parameter, ...] | None) -> str:
    """The columns an alias would expose, written off an example alias."""
    names = ", ".join(f"t.{column.name}" for column in columns or ())
    return names or "its columns"


def _type_name(node: exp.Expr | None) -> str | None:
    """The dialect type `node` spells, or None if it spells none."""
    if not isinstance(node, exp.DataType):
        return None
    kind = node.this
    if kind is exp.DataType.Type.ARRAY:
        inner = node.expressions
        if len(inner) != 1:
            return None
        element = _type_name(inner[0])
        return None if element is None or is_array(element) else f"{element}[]"
    if kind is exp.DataType.Type.USERDEFINED:
        return _ident_name(node.args.get("kind"))
    if node.expressions:  # a parameterized spelling, e.g. decimal(5, 2)
        return None
    return _SCALAR_TYPES.get(kind)


def _written_type(node: exp.Expr | None) -> str:
    """A declared type as the DIALECT spells it, else as it was written.

    sqlglot's own name for a type is not the dialect's -- ``number`` parses to
    ``DECIMAL`` -- and a message naming the wrong one sends the reader looking
    for a type this dialect does not have.
    """
    return _type_name(node) or _written(node).lower()


def _struct_fields(node: exp.Expr | None) -> list[exp.ColumnDef] | None:
    """The ``<name> <type>`` list a ``STRUCT(...)`` spells, or None if it is not one."""
    if not isinstance(node, exp.DataType) or node.this is not exp.DataType.Type.STRUCT:
        return None
    return [f for f in node.expressions if isinstance(f, exp.ColumnDef)]


def _annotation(
    node: exp.Expr | None, column: str, name: str, anchor: exp.Expr
) -> Annotation | None:
    """The record shape a ``STRUCT(...)[]`` type declares, or None for another type.

    A struct array with a field the dialect has no value type for is a
    rejection rather than a None: the writer meant an annotation column, and
    saying "unknown type" instead would name the wrong mistake.

    ``cue[]`` is the one shorthand: it declares the cue record's own shape,
    and reads back as the record it stands for.
    """
    if not isinstance(node, exp.DataType) or node.this is not exp.DataType.Type.ARRAY:
        return None
    inner = node.expressions[0] if len(node.expressions) == 1 else None
    if _type_name(inner if isinstance(inner, exp.Expr) else None) == CUE_TYPE:
        return Annotation(name=column, fields=_cue_annotation_fields())
    fields = _struct_fields(inner if isinstance(inner, exp.Expr) else None)
    if fields is None:
        return None
    declared: list[AnnotationField] = []
    for field_node in fields:
        if not isinstance(field_node.this, exp.Identifier):
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"function '{name}' declares '{column}' with an unnamed field",
                anchor,
                hint=_ANNOTATION_FIELD_HINT,
            )
        written = _ident_name(field_node.this)
        field_type = _type_name(field_node.args.get("kind"))
        if field_type not in _ANNOTATION_FIELD_TYPES:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"function '{name}' declares the field '{written}' of '{column}' "
                f"as '{_written_type(field_node.args.get('kind'))}'",
                anchor,
                hint=_ANNOTATION_FIELD_HINT,
            )
        if any(seen.name == written for seen in declared):
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"function '{name}' declares the field '{written}' of '{column}' twice",
                anchor,
                hint="one name, one field",
            )
        declared.append(AnnotationField(written, field_type))
    if not declared:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"function '{name}' declares '{column}' with no fields",
            anchor,
            hint=_ANNOTATION_FIELD_HINT,
        )
    return Annotation(name=column, fields=tuple(declared))


def _checked_type(node: exp.Expr | None, name: str, anchor: exp.Expr) -> str:
    """The type `node` declares, rejected by name if the dialect has no such type."""
    declared = _type_name(node)
    if declared is None or element_type(declared) not in NAMEABLE_TYPES:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"function '{name}' declares an unknown type '{_written(node).lower()}'",
            anchor,
            hint=_TYPE_HINT,
        )
    return declared


def _function_name(create: exp.Create) -> tuple[exp.Expr, exp.Identifier, list[exp.Expr]]:
    """The signature node, the name identifier, and the parameter list."""
    signature = create.this
    if isinstance(signature, exp.UserDefinedFunction):
        table, params = signature.this, list(signature.expressions)
    else:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            "a function is missing its parameter list",
            signature if isinstance(signature, exp.Expr) else None,
            fallback=create,
            hint=_FUNCTION_HINT,
        )
    if not isinstance(table, exp.Table):
        raise _error(
            ErrorCode.UNSUPPORTED_SQL, "a function is missing its name", create, hint=_FUNCTION_HINT
        )
    if table.args.get("db") or table.args.get("catalog"):
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            "qualified function names are not supported",
            table,
            fallback=create,
            hint="a function lives in one script, not in a schema",
        )
    identifier = table.this
    if not isinstance(identifier, exp.Identifier):
        raise _error(
            ErrorCode.UNSUPPORTED_SQL, "a function is missing its name", create, hint=_FUNCTION_HINT
        )
    return signature, identifier, params


def _properties(
    create: exp.Create, name: str, anchor: exp.Expr
) -> tuple[exp.ReturnsProperty, str]:
    """The ``RETURNS`` property and the declared language; anything else is rejected."""
    returns: exp.ReturnsProperty | None = None
    language = ""
    properties = create.args.get("properties")
    if isinstance(properties, exp.Properties):
        for prop in properties.expressions:
            if isinstance(prop, exp.ReturnsProperty):
                returns = prop
                continue
            if isinstance(prop, exp.LanguageProperty):
                language = _ident_name(prop.this) if isinstance(prop.this, exp.Expr) else ""
                continue
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"unsupported CREATE FUNCTION option: {_written(prop)}",
                anchor,
                hint="a function carries a signature, a body and its LANGUAGE, "
                "and nothing else",
            )
    if returns is None:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"function '{name}' declares no RETURNS type",
            anchor,
            hint=_FUNCTION_HINT,
        )
    return returns, language


def _body_text(create: exp.Create, name: str, anchor: exp.Expr) -> str:
    """The body as written, from either quoting Postgres allows."""
    body = create.args.get("expression")
    if isinstance(body, exp.Heredoc) and isinstance(body.this, str):
        return body.this
    if isinstance(body, exp.Literal) and body.is_string:
        return str(body.this)
    raise _error(
        ErrorCode.UNSUPPORTED_SQL,
        f"function '{name}' has no body",
        anchor,
        fallback=create,
        hint=_FUNCTION_HINT,
    )


def _reanchor(err: FfrwdError, name: str, anchor: exp.Expr) -> FfrwdError:
    """A body-static rejection, said in the script's own coordinates."""
    line, col = _pos(anchor)
    at = f" at body line {err.line}" if err.line is not None else ""
    return FfrwdError(
        err.code,
        f"the body of {name}(){at}: {err.message}",
        line=line,
        col=col,
        hint=err.hint,
    )


def _body_select(
    text: str, name: str, anchor: exp.Expr, columns: tuple[Parameter, ...] | None
) -> exp.Select:
    """Parse and shape-check one body: a single SELECT of the declared width.

    A value's body is one column; a table's is one per ``RETURNS TABLE`` name.
    """
    wanted = 1 if columns is None else len(columns)
    shape = _VALUE_BODY_HINT if columns is None else _TABLE_BODY_HINT
    try:
        parsed = parse(text)
    except FfrwdError as err:
        raise _reanchor(err, name, anchor) from err
    if not isinstance(parsed, exp.Select):
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"the body of {name}() is not one SELECT",
            anchor,
            hint=shape,
        )
    if parsed.args.get("with_") is not None:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"the body of {name}() may not have its own WITH",
            anchor,
            hint="a function body is inlined into the query that calls it; "
            "put the CTE there",
        )
    try:
        _check_query_args(
            parsed, frozenset({"expressions", "from_", "joins", "where"}), "function body"
        )
    except FfrwdError as err:
        raise _reanchor(err, name, anchor) from err
    # The metadata map is not a declared column: it is an assertion about the
    # body's own streams, so it does not count against the RETURNS TABLE arity.
    written = sum(
        1
        for projection in parsed.expressions
        if not (
            isinstance(projection, exp.Expr)
            and columns is not None
            and _projection_alias(projection) == TAGS_COLUMN
        )
    )
    if written != wanted:
        plural = "" if written == 1 else "s"
        said = (
            "and a value is one column"
            if columns is None
            else f"but its RETURNS TABLE declares {wanted}"
        )
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"the body of {name}() selects {written} column{plural}, {said}",
            anchor,
            hint=shape,
        )
    return parsed


def _body_aliases(
    body: exp.Select, name: str, params: Sequence[Parameter], anchor: exp.Expr
) -> frozenset[str]:
    """The names the body binds in its own FROM, checked against the signature."""
    aliases: set[str] = set()
    for item, _ in from_entries(body):
        alias = item.args.get("alias")
        if isinstance(alias, exp.TableAlias) and isinstance(alias.this, exp.Identifier):
            aliases.add(_ident_name(alias.this))
    shadowed = aliases.intersection(param.name for param in params)
    if shadowed:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"the body of {name}() shadows the parameter "
            f"'{sorted(shadowed)[0]}' with a FROM alias",
            anchor,
            hint="rename the alias; a parameter and a FROM item cannot share a name",
        )
    return frozenset(aliases)


def _check_body_scope(
    body: exp.Select,
    name: str,
    params: Sequence[Parameter],
    aliases: frozenset[str],
    anchor: exp.Expr,
) -> None:
    """Every name the body reads is one of its parameters or one of its own aliases."""
    known = aliases.union(param.name for param in params)
    for column in body.find_all(exp.Column):
        key = _leftmost(column)
        if key is None:
            continue
        read = _ident_name(column.args.get(key))
        if read in known:
            continue
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"the body of {name}() references '{read}', which is neither a "
            "parameter nor one of its own FROM items",
            anchor,
            hint=_BODY_SCOPE_HINT,
        )


@dataclass(frozen=True)
class _Declared:
    """What a ``<name> <type>`` list is called, and how each rejection advises."""

    noun: str
    shape: str
    plain: str
    once: str
    allow_default: bool = False
    allow_annotation: bool = False


_PARAMETER = _Declared(
    noun="parameter",
    shape=_FUNCTION_HINT,
    plain="a parameter is a name, a type, and an optional DEFAULT: no OUT, "
    "no INOUT, no VARIADIC, no COLLATE",
    once="one name, one position",
    allow_default=True,
)
# A wasm signature reads the same way, plus the annotation column a module
# that consumes rows takes.
_WASM_PARAMETER = replace(_PARAMETER, shape=_WASM_HINT, allow_annotation=True)
_TABLE_COLUMN = _Declared(
    noun="RETURNS TABLE column",
    shape=_TABLE_HINT,
    plain="a RETURNS TABLE column is a name and a type, and nothing else",
    once="one name, one column",
)


def _default_constraint(node: exp.ColumnDef) -> exp.Expr | None:
    """The literal a lone ``DEFAULT`` constraint declares, or None with none such."""
    constraints = node.args.get("constraints") or []
    if len(constraints) != 1:
        return None
    only = constraints[0]
    inner = only.args.get("kind") if isinstance(only, exp.ColumnConstraint) else None
    return inner.this if isinstance(inner, exp.DefaultColumnConstraint) else None


def _checked_default(
    default: exp.Expr, declared_type: str, name: str, written: str, anchor: exp.Expr
) -> exp.Expr:
    """The DEFAULT literal, rejected if it does not match the parameter's type.

    ``DEFAULT NULL`` is legal and matches every type: it is the spelling for
    an optional parameter that is simply absent when omitted -- the NULL flows
    down and drops wherever the body uses it, like any other NULL. Without a
    DEFAULT, omission is an arity error, so the two spellings are not
    redundant.
    """
    if isinstance(default, exp.Null):
        return default
    kind = _argument_kind(default)
    if kind is None or kind != _declared_kind(declared_type):
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"function '{name}' declares the parameter '{written}' with a "
            f"DEFAULT that is not {declared_type}",
            anchor,
            hint="a DEFAULT is a literal of the parameter's own type",
        )
    return default


def _column_defs(
    nodes: Sequence[exp.Expr], name: str, anchor: exp.Expr, create: exp.Create, kind: _Declared
) -> tuple[Parameter, ...]:
    """One ``<name> <type>`` list -- a signature's, or a ``RETURNS TABLE``'s."""
    declared: list[Parameter] = []
    seen_default = False
    for node in nodes:
        if not isinstance(node, exp.ColumnDef) or not isinstance(node.this, exp.Identifier):
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"function '{name}' has a {kind.noun} with no name",
                anchor,
                fallback=create,
                hint=kind.shape,
            )
        written = _ident_name(node.this)
        # OUT/INOUT, VARIADIC and COLLATE all land here; DEFAULT lands here too
        # unless `kind` allows it.
        constraints = node.args.get("constraints")
        default = _default_constraint(node) if kind.allow_default else None
        if constraints and default is None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"function '{name}' writes the {kind.noun} '{written}' with "
                f"{_written(constraints[0])}, which is not supported",
                anchor,
                fallback=create,
                hint=kind.plain,
            )
        if any(seen.name == written for seen in declared):
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"function '{name}' declares the {kind.noun} '{written}' twice",
                anchor,
                fallback=create,
                hint=kind.once,
            )
        annotation = (
            _annotation(node.args.get("kind"), written, name, anchor)
            if kind.allow_annotation
            else None
        )
        if annotation is not None:
            # DEFAULT NULL makes the column optional: a call over a plain
            # stream wires no rows in. Any other default would be a value,
            # and rows are not values.
            if default is not None and not isinstance(default, exp.Null):
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"function '{name}' gives the {kind.noun} '{written}' a DEFAULT",
                    anchor,
                    fallback=create,
                    hint="an annotation column is produced by the call that "
                    "fills it; DEFAULT NULL, the only default it can carry, "
                    "makes it optional",
                )
            declared.append(Parameter(written, annotation.written, default, annotation))
            continue
        declared_type = _checked_type(node.args.get("kind"), name, anchor)
        if default is not None:
            default = _checked_default(default, declared_type, name, written, anchor)
            seen_default = True
        elif seen_default:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"function '{name}' declares the {kind.noun} '{written}' with no "
                "DEFAULT after one that has one",
                anchor,
                fallback=create,
                hint="every parameter after the first DEFAULT must have one too",
            )
        declared.append(Parameter(written, declared_type, default))
    return tuple(declared)


def _table_columns(
    prop: exp.ReturnsProperty, name: str, anchor: exp.Expr, create: exp.Create
) -> tuple[Parameter, ...]:
    """The ``RETURNS TABLE(...)`` column list: the shape the call site's alias exposes."""
    schema = prop.this
    nodes = list(schema.expressions) if isinstance(schema, exp.Schema) else []
    if not nodes:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"function '{name}' declares RETURNS TABLE with no columns",
            anchor,
            fallback=create,
            hint=_TABLE_HINT,
        )
    declared = _column_defs(nodes, name, anchor, create, _TABLE_COLUMN)
    for column in declared:
        if column.name != TAGS_COLUMN:
            continue
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"function '{name}' declares a column called '{TAGS_COLUMN}'",
            anchor,
            fallback=create,
            hint=f"'{TAGS_COLUMN}' names the metadata map, which the body writes "
            f"directly (STRUCT('Main' AS title) AS {TAGS_COLUMN}); a declared "
            "column is a stream or a value",
        )
    return declared


def _module_export(
    body: exp.Expr | None, name: str, anchor: exp.Expr, create: exp.Create
) -> tuple[str, str, exp.Expr]:
    """The module path, the export name, and the node to anchor on."""
    if not isinstance(body, ModuleExport):
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"function '{name}' says LANGUAGE wasm but names no module and export",
            anchor,
            fallback=create,
            hint=_WASM_HINT,
        )
    module, export = body.this, body.expression
    if not isinstance(module, exp.Literal) or not module.is_string:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"function '{name}' does not name its module as a string",
            module if isinstance(module, exp.Expr) else None,
            fallback=anchor,
            hint=_WASM_HINT,
        )
    if not isinstance(export, exp.Literal) or not export.is_string:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"function '{name}' does not name its export as a string",
            export if isinstance(export, exp.Expr) else None,
            fallback=anchor,
            hint=_WASM_HINT,
        )
    path = str(module.this)
    if not path.strip():
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"function '{name}' names an empty module path",
            module,
            fallback=anchor,
            hint=_WASM_HINT,
        )
    if not str(export.this).strip():
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"function '{name}' names an empty export",
            export,
            fallback=anchor,
            hint=_WASM_HINT,
        )
    return path, str(export.this), module


def _define_wasm_source(
    name: str,
    module: str,
    export: str,
    params: tuple[Parameter, ...],
    identifier: exp.Identifier,
    create: exp.Create,
) -> WasmFunction:
    """One validated ``RETURNS source`` declaration: a FROM-position row source.

    The mirror of a sink: a source produces streams and reads none, so every
    parameter is a value the module is configured with -- the same domain a
    sink's own value parameters draw from -- and a stream-typed parameter is
    refused outright, since a source has nothing for it to mean.
    """
    stream = next(
        (p for p in params if element_type(p.type) in WASM_STREAM_TYPES), None
    )
    if stream is not None:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"wasm function '{name}' declares the stream parameter '{stream.name}'",
            identifier,
            fallback=create,
            hint=_WASM_SOURCE_HINT,
        )
    line, col = _pos(identifier, create)
    return WasmFunction(
        name=name,
        module=module,
        export=export,
        params=params,
        returns=WASM_SOURCE,
        line=line,
        col=col,
    )


def _define_wasm_rows(
    name: str,
    module: str,
    export: str,
    params: tuple[Parameter, ...],
    returns: str,
    returns_rows: Annotation,
    identifier: exp.Identifier,
    create: exp.Create,
) -> WasmFunction:
    """One validated ROWS declaration: rows in, rows out, no stream anywhere.

    The one parameter is the rows the module reads and the RETURNS is the
    rows it writes, each an array of annotation records. There is no stream
    to filter and no value to configure it with, so anything else in the
    signature is refused by name; a DEFAULT has nothing to mean either,
    since a rows function exists to read the rows it is handed.
    """
    if not params:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"wasm function '{name}' returns rows and reads none",
            identifier,
            fallback=create,
            hint=_WASM_ROWS_HINT,
        )
    if len(params) > 1:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"wasm function '{name}' returns rows and takes {len(params)} parameters",
            identifier,
            fallback=create,
            hint=_WASM_ROWS_HINT,
        )
    column = params[0]
    if column.annotation is None:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"wasm function '{name}' returns rows and takes '{column.name}' as "
            f"{column.type}",
            identifier,
            fallback=create,
            hint=_WASM_ROWS_HINT,
        )
    if column.default is not None:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"wasm function '{name}' gives the row column '{column.name}' a DEFAULT",
            identifier,
            fallback=create,
            hint="a rows function exists to read the rows it is handed; there "
            "is nothing to fall back to",
        )
    line, col = _pos(identifier, create)
    return WasmFunction(
        name=name,
        module=module,
        export=export,
        params=params,
        returns=returns,
        returns_rows=returns_rows,
        line=line,
        col=col,
    )


def _define_wasm_value(
    name: str,
    module: str,
    export: str,
    params: tuple[Parameter, ...],
    returns: str,
    identifier: exp.Identifier,
    create: exp.Create,
) -> WasmFunction:
    """One validated value-returning ``LANGUAGE wasm`` declaration.

    Every parameter is text, number or boolean -- the scalar domain a JSON
    Schema can hold, the same one an annotation field draws from. There is no
    stream to filter and no DEFAULT: the module runs at every call site, so
    an omitted argument has nothing compile-time to fall back to.
    """
    for param in params:
        if param.type in _ANNOTATION_FIELD_TYPES:
            if param.default is not None:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"wasm function '{name}' gives the parameter '{param.name}' "
                    "a DEFAULT",
                    identifier,
                    fallback=create,
                    hint=_WASM_VALUE_DEFAULT_HINT,
                )
            continue
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"wasm function '{name}' declares the parameter '{param.name}' as "
            f"{param.type}",
            identifier,
            fallback=create,
            hint=_WASM_VALUE_PARAM_HINT,
        )
    line, col = _pos(identifier, create)
    return WasmFunction(
        name=name,
        module=module,
        export=export,
        params=params,
        returns=returns,
        line=line,
        col=col,
    )


def _define_wasm(
    create: exp.Create,
    name: str,
    identifier: exp.Identifier,
    params: tuple[Parameter, ...],
    returns_prop: exp.ReturnsProperty,
    body: exp.Expr | None,
) -> WasmFunction:
    """One validated ``LANGUAGE wasm`` declaration: a stream filter or a value.

    A STREAM function reads one or more streams of ONE kind and writes one of
    that same kind: a leading run of ``video_stream`` or ``audio_stream``
    parameters, all the type the return names, then any number of value
    parameters that become the module's own. A module that reads annotations
    off each frame returns them beside the stream, as ``STRUCT(<stream>
    <stream type>, <name> STRUCT(...)[])``; one that consumes them takes that
    column right after its stream. Only those two shapes carry annotations,
    and every other struct is refused by name. A signature reading several
    streams may not also CONSUME annotations; returning them is unaffected.

    A VALUE function returns text, number or boolean and takes only
    parameters of those same three types -- the domain a JSON Schema can
    hold. It filters no stream: the module runs once per call, at compile
    time, on the folded arguments (:mod:`ffrwd.lower`). A wasm function
    returning a table is refused outright rather than guessed at.

    A SINK function -- ``RETURNS sink`` -- reads its streams like any
    consumer and writes nothing back: it is a COPY destination, and its
    signature carries every consumer rule except the one matching the return
    to the first stream, since there is no return.
    """
    module, export, anchor = _module_export(body, name, identifier, create)
    if returns_prop.args.get("is_table"):
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"wasm function '{name}' returns a table",
            identifier,
            fallback=create,
            hint=_WASM_VALUE_HINT,
        )
    node = returns_prop.this if isinstance(returns_prop.this, exp.Expr) else None
    # The parameter says which kind the RETURNS is read against, so a hint
    # about the struct return names the kind the writer was already writing.
    written_stream = params[0].type if params else _WASM_STREAM
    if written_stream not in WASM_STREAM_TYPES:
        written_stream = _WASM_STREAM
    stream_field, emits, struct_stream = _wasm_struct_return(
        node, name, identifier, create, written_stream
    )
    if emits is None:
        if _type_name(node) == WASM_SINK:
            returns = WASM_SINK
        elif _type_name(node) == WASM_SOURCE:
            return _define_wasm_source(name, module, export, params, identifier, create)
        else:
            # An array of annotation records as the RETURNS is a ROWS
            # function; checked before the type itself, since a bare
            # STRUCT(...)[] has no name in the type vocabulary.
            returns_rows = _annotation(node, _ROWS_RETURN, name, identifier)
            if returns_rows is not None:
                return _define_wasm_rows(
                    name,
                    module,
                    export,
                    params,
                    _type_name(node) or returns_rows.written,
                    returns_rows,
                    identifier,
                    create,
                )
            returns = _checked_type(node, name, identifier)
            if returns in _ANNOTATION_FIELD_TYPES:
                return _define_wasm_value(
                    name, module, export, params, returns, identifier, create
                )
            if returns not in WASM_STREAM_TYPES:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"wasm function '{name}' returns {returns}",
                    identifier,
                    fallback=create,
                    hint=_WASM_VALUE_HINT,
                )
    else:
        returns = struct_stream
    # A sink alone may open with a value instead of a stream: its rows then
    # come straight off the COPY's SELECT list rather than named arguments,
    # so the leading-stream requirement below does not apply to it.
    reads_rows_from_select = returns == WASM_SINK and (
        not params or element_type(params[0].type) not in WASM_STREAM_TYPES
    )
    if not reads_rows_from_select:
        if not params:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"wasm function '{name}' takes no stream",
                identifier,
                fallback=create,
                hint=_WASM_HINT,
            )
        if element_type(params[0].type) not in WASM_STREAM_TYPES:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"wasm function '{name}' takes {params[0].type} as its first "
                f"parameter '{params[0].name}'",
                identifier,
                fallback=create,
                hint=f"the stream a module filters is its first parameter, and is "
                f"{_WASM_STREAM_HINT}",
            )
    if returns != WASM_SINK and params[0].type != returns:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"wasm function '{name}' takes {params[0].type} and returns "
            f"{returns}",
            identifier,
            fallback=create,
            hint="a module filters one kind of stream: what it takes is what "
            "it returns",
        )
    streams = _leading_streams(params)
    if returns == WASM_SINK:
        _check_sink_streams(name, streams, identifier, create)
    else:
        # An ARRAY of streams reaches here only as a first parameter that
        # does not match the return, which the check above already refused:
        # a stream function's pads are its own declaration, so there is
        # nothing for an array to mean.
        mixed = next((p for p in streams if p.type != params[0].type), None)
        if mixed is not None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"wasm function '{name}' takes '{mixed.name}' as {mixed.type} "
                f"beside {params[0].type}",
                identifier,
                fallback=create,
                hint="a module reads one kind of stream: every stream parameter "
                "is the type it returns",
            )
    for extra in params[len(streams) :]:
        if extra.annotation is not None or _declared_kind(extra.type) != "stream":
            continue
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"wasm function '{name}' takes a second stream, '{extra.name}'",
            identifier,
            fallback=create,
            hint="a module's streams come first; the parameters after them "
            "are values it is configured with",
        )
    if len(streams) > 1 or any(is_array(p.type) for p in streams):
        _reject_annotation_column(name, params, identifier, create)
    for position, extra in enumerate(params[len(streams) + 1 :], start=len(streams) + 2):
        if extra.annotation is None:
            continue
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"wasm function '{name}' takes the annotation column "
            f"'{extra.name}' in position {position}",
            identifier,
            fallback=create,
            hint=_annotation_param_hint(params[0].type),
        )
    line, col = _pos(identifier, create)
    return WasmFunction(
        name=name,
        module=module,
        export=export,
        params=params,
        returns=returns,
        emits=emits,
        stream_field=stream_field,
        line=line,
        col=col,
    )


def _reject_annotation_column(
    name: str,
    params: tuple[Parameter, ...],
    identifier: exp.Identifier,
    create: exp.Create,
) -> None:
    """A signature reading several streams takes no annotation column.

    Consuming rows and reading several streams are separate features, and one
    signature may declare one of them. RETURNING rows is not affected: the
    column leaves beside the stream however many the module read.
    """
    column = next((p for p in params if p.annotation is not None), None)
    if column is None:
        return
    raise _error(
        ErrorCode.UNSUPPORTED_SQL,
        f"wasm function '{name}' takes the annotation column "
        f"'{column.name}' beside several streams",
        identifier,
        fallback=create,
        hint=_WASM_MULTI_HINT,
    )


def _wasm_struct_return(
    node: exp.Expr | None,
    name: str,
    identifier: exp.Identifier,
    create: exp.Create,
    written_stream: str,
) -> tuple[str, Annotation | None, str]:
    """A ``RETURNS STRUCT(...)``, read as a stream and its annotation column.

    ``("", None, "")`` for a return that is not a struct at all, which the
    caller then reads as the plain stream type it always was. Every struct
    that is not exactly one stream field followed by one annotation array is a
    rejection saying what is supported. `written_stream` is the kind the
    signature's parameter named, which is the kind the rejections spell.
    """
    hint = _annotation_return_hint(written_stream)
    fields = _struct_fields(node)
    if fields is None:
        return "", None, ""
    written = [
        _ident_name(f.this) if isinstance(f.this, exp.Identifier) else "" for f in fields
    ]
    if len(fields) != 2:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"wasm function '{name}' returns a struct of {len(fields)} fields",
            identifier,
            fallback=create,
            hint=hint,
        )
    if not all(written):
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"wasm function '{name}' returns a struct with an unnamed field",
            identifier,
            fallback=create,
            hint=hint,
        )
    stream = _type_name(fields[0].args.get("kind"))
    if stream not in WASM_STREAM_TYPES:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"wasm function '{name}' returns the field '{written[0]}' as "
            f"'{_written_type(fields[0].args.get('kind'))}'",
            identifier,
            fallback=create,
            hint=hint,
        )
    annotation = _annotation(fields[1].args.get("kind"), written[1], name, identifier)
    if annotation is None:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"wasm function '{name}' returns the field '{written[1]}' as "
            f"'{_written_type(fields[1].args.get('kind'))}'",
            identifier,
            fallback=create,
            hint=hint,
        )
    assert stream is not None  # narrowed by the WASM_STREAM_TYPES check above
    return written[0], annotation, stream


def _define(create: exp.Create) -> _Function | WasmFunction:
    """One validated ``CREATE FUNCTION``, body parsed and shape-checked."""
    if create.args.get("replace"):
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            "CREATE OR REPLACE FUNCTION is not supported",
            create,
            hint="a function exists only for the length of one script; "
            "there is nothing to replace",
        )
    if create.args.get("exists"):
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            "CREATE FUNCTION IF NOT EXISTS is not supported",
            create,
            hint="a function exists only for the length of one script; "
            "it never exists already",
        )
    signature, identifier, param_nodes = _function_name(create)
    _check_query_args(
        create, frozenset({"this", "kind", "expression", "properties", "begin"}), "CREATE FUNCTION"
    )
    name = _ident_name(identifier)
    if name in _RESERVED or name.upper() in exp.FUNCTION_BY_NAME:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"'{name}' is a reserved function name",
            identifier,
            fallback=signature,
            hint="pick a name the dialect does not already use",
        )

    # The LANGUAGE first: it is what says how the signature reads, since only
    # a wasm one may carry an annotation column.
    returns_prop, language = _properties(create, name, identifier)
    if not language:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"function '{name}' declares no LANGUAGE",
            identifier,
            fallback=create,
            hint="add LANGUAGE sql for a body that is a query, or LANGUAGE wasm "
            "for one a module implements",
        )
    kind = _WASM_PARAMETER if language == _WASM else _PARAMETER
    params = _column_defs(param_nodes, name, identifier, create, kind)

    written_body = create.args.get("expression")
    if language == _WASM:
        return _define_wasm(create, name, identifier, params, returns_prop, written_body)
    if language != _SQL:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"function '{name}' is written in {language}, which the dialect "
            "does not have",
            identifier,
            fallback=create,
            hint="a body is a query (LANGUAGE sql) or a wasm module (LANGUAGE wasm)",
        )
    if isinstance(written_body, ModuleExport):
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"function '{name}' names a module and an export, but says LANGUAGE sql",
            identifier,
            fallback=create,
            hint="a module is hosted, not inlined; say LANGUAGE wasm",
        )
    columns: tuple[Parameter, ...] | None = None
    if returns_prop.args.get("is_table"):
        columns = _table_columns(returns_prop, name, identifier, create)
        returns = "TABLE(" + ", ".join(f"{c.name} {c.type}" for c in columns) + ")"
    else:
        node = returns_prop.this if isinstance(returns_prop.this, exp.Expr) else None
        returns = _checked_type(node, name, identifier)

    body = _body_select(_body_text(create, name, identifier), name, identifier, columns)
    aliases = _body_aliases(body, name, params, identifier)
    _check_body_scope(body, name, params, aliases, identifier)
    return _Function(
        name=name,
        params=params,
        returns=returns,
        body=body,
        node=identifier,
        aliases=aliases,
        position=0,
        columns=columns,
    )


def _in_lib(
    err: FfrwdError, name: str, path: Path, anchor: exp.Expr | None
) -> FfrwdError:
    """A rejection from a package's lib file, said at the call site that reached for it.

    A lib file's line numbers mean nothing in the query the reader is
    looking at, so the anchor moves to the call and the message carries the
    file and the line it really came from. With no call to point at -- a
    listing rather than a compile -- the file's own line is all there is.
    """
    line, col = _pos(anchor) if anchor is not None else (err.line, err.col)
    at = f" line {err.line}" if err.line is not None else ""
    return FfrwdError(
        err.code,
        f"package '{name}' lib {path}{at}: {err.message}",
        line=line,
        col=col,
        hint=err.hint,
    )


def _not_a_table(declared: WasmFunction, item: exp.Table) -> FfrwdError | None:
    """A wasm function written in FROM: it returns a stream or a value, never a table.

    A source is the one exception: it IS exactly a table in FROM, so it earns
    no refusal here, and None means "let it through". Binding the FROM item
    to what the source produces is the lowering half's seam
    (:mod:`ffrwd.lower`), not this module's -- past this point the call may
    still be refused downstream until that half lands.
    """
    written = declared.called
    if declared.is_source:
        return None
    if declared.is_sink:
        return _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"wasm function '{written}' returns sink, not a table",
            item,
            hint=_WASM_SINK_HINT,
        )
    if declared.is_value:
        return _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"wasm function '{written}' returns {declared.returns}, not a table",
            item,
            hint=f"call it where a value belongs: {written}(...) inside a metadata STRUCT",
        )
    return _error(
        ErrorCode.UNSUPPORTED_SQL,
        f"wasm function '{written}' returns a stream, not a table",
        item,
        hint=f"call it where a stream belongs: {written}(f.video[1]) in the SELECT list",
    )


def _package_module(
    declared: WasmFunction, package: Package, path: Path, anchor: exp.Expr | None
) -> WasmFunction:
    """`declared` with its module path read against the package's own root.

    A lib file is read out of the store or out of a linked directory, and is
    compiled from whatever working directory the caller happens to be in, so
    the written path can only mean one thing: a file the package ships. One
    that reaches outside the package root is a rejection rather than a path
    resolved against a machine the package's author never saw.
    """
    if leaves_package(declared.module):
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"package '{package.name}' lib {path}: the wasm function "
            f"'{declared.name}' names the module '{declared.module}', which "
            "leaves the package directory",
            anchor,
            hint="a package's modules ship inside it; the path is relative to "
            f"{package.root}",
        )
    return replace(
        declared,
        module=str(under_package(package.root, declared.module)),
        package=package.name,
        package_version=package.version,
    )


def _source_definitions(
    package: Package, path: Path, anchor: exp.Expr | None
) -> list[_Function | WasmFunction]:
    """Every ``CREATE FUNCTION`` one lib file holds, validated.

    A lib file is a LIBRARY, so it holds definitions and nothing else: a
    SELECT or a COPY in one is a script, and is rejected as one. Every
    definition it yields carries the package's name, which is what makes it a
    library's rather than the script's; a ``LANGUAGE wasm`` one also gets its
    module path resolved against the package root.

    `path` is always one of ``package.exports``' files. A recipe's file is a
    query and is never read here.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as err:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"package '{package.name}': could not read {path}: "
            f"{err.strerror or err}",
            anchor,
            hint=f"{package.manifest} names it in lib",
        ) from err
    try:
        statements = _statements(parse(text))
    except FfrwdError as err:
        raise _in_lib(err, package.name, path, anchor) from err

    definitions: list[_Function | WasmFunction] = []
    for statement in statements:
        if not (isinstance(statement, exp.Create) and _create_kind(statement) == "FUNCTION"):
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"package '{package.name}' lib {path} holds a statement that "
                "is not a CREATE FUNCTION",
                anchor,
                hint="a lib file is a library: it defines functions, and a "
                "query of its own is a recipe, declared in bin",
            )
        try:
            function = _define(statement)
        except FfrwdError as err:
            raise _in_lib(err, package.name, path, anchor) from err
        if isinstance(function, WasmFunction):
            definitions.append(_package_module(function, package, path, anchor))
            continue
        function.package = package.name
        function.package_version = package.version
        # Ahead of every statement of the calling script: a library is already
        # defined when the query that calls it is written.
        function.position = -1
        definitions.append(function)
    return definitions


def _package_scope(
    package: Package, anchor: exp.Expr | None
) -> tuple[dict[str, _Function | WasmFunction], dict[str, Path]]:
    """Every definition across `package`'s lib files, and the file each came from.

    One flat scope per package -- a bare call in a library body resolves here
    -- so one name defined in two of its files is a rejection.
    """
    scope: dict[str, _Function | WasmFunction] = {}
    origin: dict[str, Path] = {}
    for path in dict.fromkeys(package.exports.values()):
        for function in _source_definitions(package, path, anchor):
            if function.name in scope:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"package '{package.name}' defines '{function.name}' twice: "
                    f"{origin[function.name]} and {path}",
                    anchor,
                    hint="one name, one definition, across all of a package's lib files",
                )
            scope[function.name] = function
            origin[function.name] = path
    return scope, origin


def _check_exported(
    package: Package,
    scope: dict[str, _Function | WasmFunction],
    origin: dict[str, Path],
    anchor: exp.Expr | None,
) -> None:
    """The manifest's promise, checked: each exported name is defined in its named file."""
    for exported, path in package.exports.items():
        if exported in scope and origin[exported] == path:
            continue
        hint = (
            "a string lib's file must define a function named for the package segment"
            if exported == package.package
            else "a map lib is keyed by exported function name; the file must define "
            f"CREATE FUNCTION {exported}"
        )
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"package '{package.name}': {path} does not define '{exported}'",
            anchor,
            hint=hint,
        )


@dataclass(frozen=True)
class Signature:
    """One function a package exports: its declared shape, and from where.

    `package` is the owning package's name; `params` and `returns` are the
    declared types, not the body's; `export` is the file the definition was
    read out of.
    """

    package: str
    name: str
    params: tuple[Parameter, ...]
    returns: str
    export: Path

    @property
    def written(self) -> str:
        """The call form with its parameters: ``fn(track audio_stream)``."""
        return f"{self.name}({', '.join(_written_param(p) for p in self.params)})"


def package_signatures(package: Package) -> tuple[Signature, ...]:
    """Every function `package` exports, in the manifest's own order.

    The same reading and validation a compile does, without one: it is what
    answers "what did I just install" for a caller holding no query. Raises
    ``FfrwdError`` for a lib file that cannot be read, does not parse, holds
    anything but ``CREATE FUNCTION``, or does not define the function the
    manifest exports from it.
    """
    scope, origin = _package_scope(package, None)
    _check_exported(package, scope, origin, None)
    return tuple(
        Signature(
            package=package.name,
            name=exported,
            params=scope[exported].params,
            returns=scope[exported].returns,
            export=path,
        )
        for exported, path in package.exports.items()
    )


def package_modules(package: Package) -> tuple[WasmFunction, ...]:
    """Every ``LANGUAGE wasm`` declaration across `package`'s lib files.

    Their module paths are already resolved against the package's own root,
    so a caller holding no query -- one asking what a package's modules
    declare, or where the model an export loads belongs -- reads exactly the
    paths a compile would. Raises ``FfrwdError`` for a lib file that cannot be
    read or does not parse, like :func:`package_signatures`.
    """
    scope, _origin = _package_scope(package, None)
    return tuple(
        declared for declared in scope.values() if isinstance(declared, WasmFunction)
    )


# -- reading and rewriting nodes ------------------------------------------

# A column's qualifiers, outermost first: the arg holding the LEFTMOST written
# identifier is the alias (or parameter) the column reads.
_QUALIFIERS = ("catalog", "db", "table", "this")


def _leftmost(column: exp.Column) -> str | None:
    """The arg key holding the name `column` reads, or None for a star."""
    for key in _QUALIFIERS:
        node = column.args.get(key)
        if isinstance(node, exp.Identifier):
            return key
    return None


def _dialect_owned(segments: tuple[str, ...]) -> bool:
    """True when the dialect answers this qualifier itself, so no package may.

    ``ffmpeg`` and ``wasm`` are refused to packages outright. ``ffrwd`` is the
    official namespace and packages DO claim it, so segment count decides:
    two segments is ``ffrwd.<macro>(...)``, which is always a macro, and three
    is ``ffrwd.<package>.<member>(...)``, which is always a package. A package
    named for a macro is refused where names are validated, so the two forms
    never fight over one spelling.
    """
    if segments[0] in RESERVED_NAMESPACES:
        return True
    return len(segments) == 1 and segments[0] == MACRO_NAMESPACE


def _dot_segments(node: exp.Expr) -> tuple[str, ...] | None:
    """The plain identifiers `node` chains, left to right, or None if it is not one.

    `node` is the qualifier written before a call's parens: a bare identifier
    for a two-part call, or a `Dot` of identifiers for a three-part one --
    sqlglot parses ``a.b.c(...)`` as ``Dot(Dot(a, b), Anonymous(c))``, so a
    three-part call's qualifier is itself a `Dot`, and this walks its left
    spine to flatten it back into segments.
    """
    if isinstance(node, exp.Identifier):
        return (_ident_name(node),)
    if isinstance(node, exp.Dot) and isinstance(node.expression, exp.Identifier):
        left = _dot_segments(node.this)
        return None if left is None else (*left, _ident_name(node.expression))
    return None


def _path_after(column: exp.Column, key: str) -> list[exp.Identifier]:
    """The identifiers written to the right of `key`, in written order."""
    rest = _QUALIFIERS[_QUALIFIERS.index(key) + 1 :]
    return [
        node for name in rest if isinstance(node := column.args.get(name), exp.Identifier)
    ]


def _preorder(node: exp.Expr, *, stop: type[exp.Expr] | None = None) -> Iterator[exp.Expr]:
    """`node` and its subtree, parents first, not descending into `stop` nodes."""
    yield node
    for child in node.iter_expressions():
        if stop is not None and isinstance(child, stop):
            continue
        yield from _preorder(child, stop=stop)


def _call_name(node: object) -> str:
    """The bare name an anonymous call writes; "" for anything else.

    A namespaced call (``ffmpeg.sine()``) is an ``exp.Dot``, never this, so a
    function can never shadow one.
    """
    return str(node.name).lower() if isinstance(node, exp.Anonymous) else ""


def _series_aliases(body: exp.Expr) -> set[str]:
    """The body's ``generate_series`` aliases, each of which names its own
    one column as well as its table (``generate_series(1, 5) i`` -> ``i.i``)."""
    found: set[str] = set()
    for table in body.find_all(exp.Table):
        alias = table.args.get("alias")
        if not isinstance(table.this, exp.GenerateSeries):
            continue
        if isinstance(alias, exp.TableAlias) and isinstance(alias.this, exp.Identifier):
            found.add(_ident_name(alias.this))
    return found


def _rename(body: exp.Expr, mapping: dict[str, str]) -> None:
    """Rewrite every alias the body binds, and every reference to one."""
    series = _series_aliases(body)
    for node in body.walk():
        if isinstance(node, exp.TableAlias) and isinstance(node.this, exp.Identifier):
            replacement = mapping.get(_ident_name(node.this))
            if replacement is not None:
                node.this.set("this", replacement)
        elif isinstance(node, exp.Column):
            key = _leftmost(node)
            if key is None:
                continue
            identifier = node.args.get(key)
            if not isinstance(identifier, exp.Identifier):
                continue
            name = _ident_name(identifier)
            replacement = mapping.get(name)
            if replacement is None:
                continue
            identifier.set("this", replacement)
            if key == "this" or name not in series:
                continue
            # A series alias names its column too, so `i.i` moves whole.
            own = node.args.get("this")
            if isinstance(own, exp.Identifier) and _ident_name(own) == name:
                own.set("this", replacement)


def _accessor(argument: exp.Expr, path: list[exp.Identifier]) -> exp.Expr:
    """`path` read off `argument`, in the shape the same query written by hand takes.

    Over a plain column the dialect spells that as one dotted column
    (``t.tags.language``); over anything else -- a subscript, a filter call --
    as the parenthesized accessor (``(f.audio[1]).codec``).
    """
    if isinstance(argument, exp.Column):
        written = [
            part
            for key in _QUALIFIERS
            if isinstance(part := argument.args.get(key), exp.Identifier)
        ]
        joined = [*written, *path]
        if len(joined) <= len(_QUALIFIERS):
            keys = _QUALIFIERS[len(_QUALIFIERS) - len(joined) :]
            return exp.Column(**{key: part.copy() for key, part in zip(keys, joined)})
    read: exp.Expr = exp.Paren(this=copy.deepcopy(argument))
    for part in path:
        read = exp.Dot(this=read, expression=part.copy())
    return read


def _substitute(body: exp.Select, bindings: dict[str, exp.Expr]) -> None:
    """Replace every parameter reference with the argument bound to it.

    A bare reference becomes the argument itself; a reference with a path off
    it (``track.tags.language``) becomes the accessor form over the argument,
    which is what the same query written by hand parses to.
    """
    for column in list(body.find_all(exp.Column)):
        key = _leftmost(column)
        if key is None:
            continue
        argument = bindings.get(_ident_name(column.args.get(key)))
        if argument is None:
            continue
        if key == "this":
            column.replace(copy.deepcopy(argument))
            continue
        column.replace(_accessor(argument, _path_after(column, key)))


def _and_into(host: exp.Select, predicate: exp.Expr) -> None:
    """Add one conjunct to the host query's WHERE."""
    conjunct = exp.Paren(this=predicate) if isinstance(predicate, exp.Or) else predicate
    where = host.args.get("where")
    if isinstance(where, exp.Where) and isinstance(where.this, exp.Expr):
        where.set("this", exp.And(this=where.this, expression=conjunct))
        return
    host.set("where", exp.Where(this=conjunct))


def _splice(host: exp.Select, body: exp.Select) -> None:
    """Move the body's FROM items and WHERE into the query being compiled."""
    added: list[exp.Join] = []
    from_ = body.args.get("from_")
    if isinstance(from_, exp.From) and isinstance(from_.this, exp.Expr):
        if host.args.get("from_") is None:
            host.set("from_", exp.From(this=from_.this))
        else:
            added.append(exp.Join(this=from_.this))
    added.extend(join for join in body.args.get("joins") or [] if isinstance(join, exp.Join))
    if added:
        host.set("joins", [*(host.args.get("joins") or []), *added])
    where = body.args.get("where")
    if isinstance(where, exp.Where) and isinstance(where.this, exp.Expr):
        _and_into(host, where.this)


def _name_columns(
    body: exp.Select, columns: tuple[Parameter, ...], name: str, anchor: exp.Expr
) -> None:
    """Alias each projection to the column ``RETURNS TABLE`` named for it, in order.

    Naming the projections is what makes the generated CTE expose the declared
    columns: a CTE exposes what its body wrote ``AS``, and nothing else. The
    declared TYPE is checked here too: a stream column has to be written as a
    stream and a value column as a value, so what a caller reads off the alias
    is what the signature promised.

    A projection the body already aliased ``tags`` is not one of them: the
    metadata map is an assertion about the body's own streams, not a column of
    its rows, so it keeps its name, rides the streams the call contributes, and
    is skipped when the declared columns are handed out in order.
    """
    named: list[exp.Expr] = []
    declared = iter(columns)
    for projection in body.expressions:
        if not isinstance(projection, exp.Expr):
            continue
        if _projection_alias(projection) == TAGS_COLUMN:
            named.append(projection)
            continue
        column = next(declared, None)
        if column is None:
            break
        inner = projection.this if isinstance(projection, exp.Alias) else projection
        if isinstance(inner, exp.Expr):
            _check_column_kind(inner, column, name, anchor)
        named.append(
            exp.Alias(this=inner, alias=exp.Identifier(this=column.name, quoted=False))
        )
    body.set("expressions", named)


def _projection_alias(projection: exp.Expr) -> str | None:
    """The name a body projection was written ``AS``, folded, else None."""
    if not isinstance(projection, exp.Alias):
        return None
    alias = projection.args.get("alias")
    return _ident_name(alias) if isinstance(alias, exp.Expr) else None


def _check_column_kind(
    projection: exp.Expr, column: Parameter, name: str, anchor: exp.Expr
) -> None:
    """One body projection against the column type ``RETURNS TABLE`` declares.

    What is checked is stream-ness, which is the whole difference a caller
    sees: a stream column becomes an output, a value column becomes a value
    its rows carry. A shape that says nothing (a bare NULL, a CASE) is left to
    lowering, which sees the values.
    """
    declared_stream = _declared_kind(column.type) == "stream"
    written_stream = _writes_stream(projection)
    if written_stream is None or written_stream == declared_stream:
        return
    if declared_stream:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"column '{column.name}' of '{name}' is declared {column.type} but "
            "its body writes a value",
            projection,
            fallback=anchor,
            hint=f"select a stream for {column.name} -- the row itself names "
            f"one -- or declare {column.name} text or number",
        )
    raise _error(
        ErrorCode.UNSUPPORTED_SQL,
        f"column '{column.name}' of '{name}' is declared {column.type} but its "
        "body writes a stream",
        projection,
        fallback=anchor,
        hint=f"declare {column.name} a stream type, e.g. video_stream, or "
        f"select a value for {column.name}",
    )


def _writes_stream(node: exp.Expr) -> bool | None:
    """Whether a body projection's SHAPE makes it a stream, or None if open.

    A bare alias IS its row's stream and a subscript or filter call names one;
    a qualified column is that row's metadata, except the stream ARRAYS an
    input carries.
    """
    if isinstance(node, exp.Paren) and isinstance(node.this, exp.Expr):
        return _writes_stream(node.this)
    if isinstance(node, exp.Bracket):
        return True
    if isinstance(node, exp.Column):
        if node.args.get("table") is None:
            return True
        return _ident_name(node.this) in STREAM_ARRAY_COLUMNS
    kind = _argument_kind(node)
    return None if kind is None else kind == "stream"


def _query_node(statement: exp.Expr) -> exp.Expr | None:
    """The node whose ``WITH`` holds a statement's CTEs, read as the resolver reads it."""
    node: object = statement
    if isinstance(statement, exp.Copy):
        node = statement.this
    elif isinstance(statement, exp.Create):
        node = statement.args.get("expression")
    while isinstance(node, exp.Subquery | exp.Paren) and isinstance(
        node.this, exp.Select | exp.Union
    ):
        node = node.this
    return node if isinstance(node, exp.Select | exp.Union) else None


def _containing_cte(node: exp.Expr) -> exp.Expr | None:
    """The CTE `node` is written inside, if any."""
    parent = node.parent
    while parent is not None:
        if isinstance(parent, exp.CTE):
            return parent
        parent = parent.parent
    return None


@dataclass(frozen=True)
class _Site:
    """Where a generated CTE goes: which query's ``WITH``, and before which entry.

    A CTE only sees the CTEs written before it, so a call inside a CTE body
    puts its own CTE ahead of that one; a call in the main query appends.
    """

    query: exp.Expr
    before: exp.Expr | None


@dataclass(frozen=True)
class _CallSite:
    """One call to expand: which definition, its arguments, and what it replaces.

    `node` is the node expansion swaps out, which is not always the call: a
    namespaced value call is the ``exp.Dot`` around it, and a row source is
    the whole ``exp.Table`` FROM item.
    """

    function: _Function
    call: exp.Anonymous
    node: exp.Expr

    @property
    def row_source(self) -> bool:
        return isinstance(self.node, exp.Table)


@dataclass(frozen=True)
class _WasmSite:
    """One call to a package's ``LANGUAGE wasm`` function.

    There is no body to inline, so this is not an expansion: the call is
    rewritten to name a declaration the script now holds, and lowering
    resolves it exactly as it resolves one the script wrote itself.
    """

    declared: WasmFunction
    call: exp.Anonymous
    node: exp.Expr


# -- argument types --------------------------------------------------------


def _argument_kind(
    node: exp.Expr, wasm: Mapping[str, WasmFunction] | None = None
) -> str | None:
    """What the argument's shape says its type is, or None if it says nothing.

    `wasm` is the declarations in scope, keyed by call name -- a call naming
    one of them classifies by what IT returns, a value function folding to
    its own RETURNS type, rather than every bare call being assumed a filter.
    None where no such scope applies (a DEFAULT, a table function's body),
    which keeps every other call the stream it always was.
    """
    if isinstance(node, exp.Paren) and isinstance(node.this, exp.Expr):
        return _argument_kind(node.this, wasm)
    if isinstance(node, exp.Literal):
        return "text" if node.is_string else "number"
    for shapes, kind in _ARGUMENT_KINDS:
        if isinstance(node, shapes):
            return kind
    # A filter call, bare or namespaced: its output is a stream, unless it
    # names a wasm function whose own RETURNS says otherwise. Deliberately
    # not every exp.Func -- CASE and CAST are Funcs too, and are values.
    if isinstance(node, exp.Anonymous):
        return _wasm_argument_kind(node, wasm)
    expression = node.args.get("expression") if isinstance(node, exp.Dot) else None
    if isinstance(expression, exp.Anonymous):
        return _wasm_argument_kind(expression, wasm)
    if isinstance(expression, exp.Func):
        return "stream"
    return None


def _wasm_argument_kind(call: exp.Anonymous, wasm: Mapping[str, WasmFunction] | None) -> str:
    """A call's kind, by the RETURNS of the wasm function it names, else a stream."""
    declared = wasm.get(_call_name(call)) if wasm is not None else None
    return _declared_kind(declared.returns) if declared is not None else "stream"


def _writes_annotation(
    declared: WasmFunction,
    arguments: Sequence[exp.Expr],
    wasm: Mapping[str, WasmFunction],
) -> bool:
    """Whether this call writes its annotation column in the declared position."""
    at = declared.stream_arity
    if declared.reads is None or len(arguments) <= at:
        return False
    return is_annotation_argument(arguments[at], wasm)


def _qualified_by(call: exp.Anonymous) -> bool:
    """True when a name before the dot owns this call, as in ``ffmpeg.sine()``.

    A call on the OTHER side of a dot is not qualified: it is the call a field
    is read off, and it resolves by its own name.
    """
    parent = call.parent
    return isinstance(parent, exp.Dot) and parent.expression is call


def _annotation_return_hint(stream: str) -> str:
    """The one STRUCT return shape that is wired, spelled for one stream kind.

    The stream the module filtered, and the annotations it read off each frame.
    """
    return (
        f"write RETURNS STRUCT(<stream> {stream}, <name> STRUCT(<field> "
        f"<type>, ...)[]) -- one stream field and one annotation array, in that "
        f"order; {CUE_TYPE}[] is short for the cue record"
    )


def _annotation_param_hint(stream: str) -> str:
    """Where a module that reads annotations declares them."""
    return (
        "a module that reads annotations takes them right after its stream: "
        f"(<stream> {stream}, <name> STRUCT(<field> <type>, ...)[])"
    )


def _declared_kind(declared: str) -> str:
    """What an argument of the declared type has to look like."""
    return "stream" if TYPES[element_type(declared)].kind != "scalar" else declared


def _is_null(node: exp.Expr) -> bool:
    """True for a written NULL, through any wrapping parens."""
    if isinstance(node, exp.Paren) and isinstance(node.this, exp.Expr):
        return _is_null(node.this)
    return isinstance(node, exp.Null)


# -- the pass --------------------------------------------------------------


@dataclass
class _Expander:
    """One script's worth of definitions, call sites and inlinings."""

    functions: dict[str, _Function] = field(default_factory=dict)
    # The LANGUAGE wasm declarations, which have no body to inline: their
    # calls are checked here and left for lowering to resolve. The script's
    # own, plus every package one a call adopted.
    wasm: dict[str, WasmFunction] = field(default_factory=dict)
    wasm_used: set[str] = field(default_factory=set)
    expansions: list[_Expansion] = field(default_factory=list)
    taken: set[str] = field(default_factory=set)
    budget: int = _EXPANSION_BUDGET
    # Where the statement being walked keeps its generated CTEs.
    site: _Site | None = None
    # The packages a qualified call may resolve in, and -- keyed by (package
    # name, version), since two versions of one name are never the same
    # scope -- what each one's lib files define and which of those names its
    # manifest exports, once they have been read.
    packages: PackageSet | None = None
    scopes: dict[tuple[str, str], dict[str, _Function | WasmFunction]] = field(
        default_factory=dict
    )
    exported: dict[tuple[str, str], frozenset[str]] = field(default_factory=dict)
    # The name each package wasm declaration was adopted into the script under,
    # keyed by (package name, version, member): one declaration per member,
    # however many calls reach it.
    adopted: dict[tuple[str, str, str], str] = field(default_factory=dict)
    # Where a diagnostic that is not a rejection goes; None is silence.
    on_warning: OnWarning | None = None
    # Whose definitions a BARE call name sees: the (name, version) of the
    # package whose body is expanding, or None for the script's own.
    scope: tuple[str, str] | None = None
    # The (name, version) of the package the script ships in -- a recipe
    # compiled by name -- or None for inline SQL and -f. Only qualified
    # calls read it: they resolve at the versions this package declares.
    owner: tuple[str, str] | None = None

    # -- entry point ------------------------------------------------------

    def run(self, tree: exp.Expr) -> exp.Expr:
        """`tree` with the definitions lifted out and every call inlined."""
        statements = _statements(tree)
        rest = self._collect(statements)
        if not self.functions and not self.wasm and not self._claimed:
            return tree
        self.taken = {
            _ident_name(node)
            for statement in rest
            for node in statement.walk()
            if isinstance(node, exp.Identifier)
        }
        for position, statement in enumerate(rest):
            self._expand_statement(statement, position)
        # After inlining: a package sink call in TO position is a bare name
        # by now, so the rewrite reads script and package sinks the same way.
        for statement in rest:
            self._rewrite_sink_copy(statement)
        # After inlining: a wasm call may have been written inside a body, and
        # is only in the script once that body is spliced in.
        for position, statement in enumerate(rest):
            self._check_wasm_calls(statement, position)
        unused = self._uncalled()
        if unused is not None:
            line, col = (
                (unused.line, unused.col)
                if isinstance(unused, WasmFunction)
                else _pos(unused.node)
            )
            raise FfrwdError(
                ErrorCode.UNSUPPORTED_SQL,
                f"function '{unused.name}' is never called",
                line=line,
                col=col,
                hint="every function must be called by a later view or COPY; "
                "check the spelling of the name at its call sites",
            )
        if len(rest) == 1:
            return rest[0]
        tree.set("expressions", rest)
        return tree

    def _uncalled(self) -> _Function | WasmFunction | None:
        """The first definition that had to be called and was not.

        The script-versus-library asymmetry, in one place: a definition in a
        package EXPORT is a library's, and a library exports more than any one
        query calls, so only the script's own definitions carry the rule --
        a package's wasm declaration included, which is only ever in `wasm`
        because a call reached it.
        """
        for function in (*self.functions.values(), *self._exported()):
            if isinstance(function, WasmFunction) or function.library or function.used:
                continue
            return function
        for declared in self.wasm.values():
            if declared.position < 0:  # adopted, so already called
                continue
            if declared.name not in self.wasm_used:
                return declared
        return None

    def _exported(self) -> Iterator[_Function | WasmFunction]:
        """Every definition read out of a package's lib files so far."""
        for functions in self.scopes.values():
            yield from functions.values()

    def _collect(self, statements: list[exp.Expr]) -> list[exp.Expr]:
        """Read every definition out of the script; return what is left to compile."""
        rest: list[exp.Expr] = []
        written = False
        for statement in statements:
            if isinstance(statement, exp.Create) and _create_kind(statement) == "FUNCTION":
                if written:
                    raise _error(
                        ErrorCode.UNSUPPORTED_SQL,
                        "a CREATE FUNCTION may not follow a COPY",
                        statement,
                        hint="define every function before the first COPY",
                    )
                function = _define(statement)
                if function.name in self.functions or function.name in self.wasm:
                    line, col = (
                        (function.line, function.col)
                        if isinstance(function, WasmFunction)
                        else _pos(function.node)
                    )
                    raise FfrwdError(
                        ErrorCode.UNSUPPORTED_SQL,
                        f"function '{function.name}' is defined twice",
                        line=line,
                        col=col,
                        hint="one name, one signature; functions are not overloaded",
                    )
                if isinstance(function, WasmFunction):
                    self.wasm[function.name] = replace(function, position=len(rest))
                    continue
                function.position = len(rest)
                self.functions[function.name] = function
                continue
            written = written or isinstance(statement, exp.Copy)
            rest.append(statement)
        return rest

    # -- packages ---------------------------------------------------------

    @property
    def _claimed(self) -> bool:
        """True when some package claims a namespace this compile may resolve in."""
        return self.packages is not None and bool(self.packages.packages)

    @contextmanager
    def _scoped(self, scope: tuple[str, str] | None) -> Iterator[None]:
        """Whose definitions a bare call name sees while a function's body expands.

        A package's lib files are its own flat namespace: a library body
        calling ``helper()`` means the package's own helper, never the
        script's, and a script body never sees a package's. `scope` is the
        (name, version) of the owning package, or None for the script's own
        -- two versions of one package name are two scopes, never one
        conflated by name alone.
        """
        previous = self.scope
        self.scope = scope
        try:
            yield
        finally:
            self.scope = previous

    def _visible(self, name: str) -> _Function | WasmFunction | None:
        """The definition a BARE call name resolves to where it is written."""
        if self.scope is not None:
            return self.scopes.get(self.scope, {}).get(name)
        return self.functions.get(name)

    def _member(
        self, segments: tuple[str, ...], call: exp.Anonymous, anchor: exp.Expr
    ) -> _Function | WasmFunction | None:
        """The definition a qualified call names, or None if there is no project.

        `segments` is the identifiers written before the call name: one for a
        two-part call (``ns.pkg(...)``), two for a three-part one
        (``ns.pkg.member(...)``). None keeps a qualified call exactly what it
        was before packages existed: outside a project, ``me.pick(...)`` is not
        a package call and the rejection it earns downstream is the one it has
        always earned.

        Two-part resolution is `namespace.package(...)`, reaching the
        package's default export -- there is no alias to try first, since a
        call across packages is always written in full. Three-part resolution
        is `namespace.package.member` outright.
        """
        packages = self.packages
        if packages is None or not packages.packages:
            return None
        name = _call_name(call)
        if len(segments) == 1:
            (qualifier,) = segments
            package = self._package_at(qualifier, name, anchor, explicit=False)
            return self._reach(package, None, package.name.replace("/", "."), anchor)
        namespace, package_name = segments
        package = self._package_at(namespace, package_name, anchor)
        return self._reach(package, name, f"{namespace}.{package_name}", anchor)

    def _package_at(
        self, namespace: str, package_name: str, anchor: exp.Expr, *, explicit: bool = True
    ) -> Package:
        """The package `namespace.package_name` names, at the version the calling
        package (or the project, outside one) itself depends on -- or a typed
        rejection.

        An unknown namespace says what is installed; a known namespace with
        no such package says what it holds. `explicit` is False for a
        two-part call: with no alias to try first, an unknown first segment
        there is likely someone reaching for the old aliased form, so the
        hint points at the explicit one instead of guessing at a namespace
        typo.
        """
        packages = self.packages
        assert packages is not None
        # A recipe's own statements resolve at its package's versions too.
        holder = self.scope if self.scope is not None else self.owner
        dependent = holder[0] if holder is not None else None
        found = packages.resolve(dependent, f"{namespace}/{package_name}")
        if found is not None:
            return found
        known = packages.namespaces()
        if namespace not in known:
            if not explicit:
                raise _error(
                    ErrorCode.UNKNOWN_FUNCTION,
                    f"unknown namespace '{namespace}'",
                    anchor,
                    hint="calls across packages are written "
                    "<namespace>.<package>.<member>",
                )
            near = difflib.get_close_matches(namespace, list(known), n=1, cutoff=0.6)
            raise _error(
                ErrorCode.UNKNOWN_FUNCTION,
                f"unknown namespace '{namespace}'",
                anchor,
                hint=f"did you mean '{near[0]}'?"
                if near
                else (
                    f"namespaces this project can call: {', '.join(known)}"
                    if known
                    else "no packages are installed"
                ),
            )
        held = [package.package for package in packages.in_namespace(namespace)]
        raise _error(
            ErrorCode.UNKNOWN_FUNCTION,
            f"namespace '{namespace}' has no package '{package_name}'",
            anchor,
            hint=f"{namespace} holds: {', '.join(held)}"
            if held
            else f"{namespace} holds no packages",
        )

    def _reach(
        self, package: Package, member: str | None, written: str, anchor: exp.Expr
    ) -> _Function | WasmFunction:
        """The definition `package` exports at `member`, or its default at None.

        A missing default names the package's exports instead of guessing
        which one was meant -- worded one way for a package with no exports
        at all, another for one whose `lib` is a map and so names its exports
        rather than offering a default; a missing named member gets the usual
        did-you-mean.
        """
        exports = self._scope_of(package, anchor)
        identity = (package.name, package.version)
        key = package.package if member is None else member
        function = exports.get(key) if key in self.exported[identity] else None
        if function is not None:
            return function
        if member is None:
            named = sorted(self.exported[identity])
            if named:
                raise _error(
                    ErrorCode.UNKNOWN_FUNCTION,
                    f"package '{package.name}' names its exports",
                    anchor,
                    hint=f"it exports: {', '.join(named)}",
                )
            raise _error(
                ErrorCode.UNKNOWN_FUNCTION,
                f"package '{package.name}' has no default export",
                anchor,
                hint=f"{package.name} exports nothing",
            )
        exported = sorted(self.exported[identity])
        near = difflib.get_close_matches(member, exported, n=1, cutoff=0.6)
        raise _error(
            ErrorCode.UNKNOWN_FUNCTION,
            f"package '{package.name}' has no export '{member}'",
            anchor,
            hint=f"did you mean {written}.{near[0]}()?"
            if near
            else f"{package.name} exports: {', '.join(exported) or 'nothing'}",
        )

    def _warn_about(self, package: Package, anchor: exp.Expr) -> None:
        """Say what resolving in `package` cost, at the first call that reaches it.

        Once per package: this runs where a package's lib files are read, and
        they are read once per compile. Both warnings are about where the
        definition came FROM, so a query that never calls into the package
        hears nothing.
        """
        if self.on_warning is None:
            return
        line, col = _pos(anchor)
        if package.linked:
            self.on_warning(
                FfrwdWarning(
                    WarningCode.LINKED_PACKAGE,
                    package.name,
                    f"package '{package.name}' is linked to {package.root}, so this "
                    "command depends on files no lockfile pins",
                    line=line,
                    col=col,
                    hint="a linked package is edited in place; install a version of it to "
                    "get a build that can be reproduced",
                )
            )
        packages = self.packages
        if package.layer == "global" and packages is not None and packages.in_project:
            self.on_warning(
                FfrwdWarning(
                    WarningCode.GLOBAL_PACKAGE,
                    package.name,
                    f"package '{package.name}' was resolved from the machine-wide "
                    f"lockfile, not from the project at {packages.root}",
                    line=line,
                    col=col,
                    hint=f"install '{package.name}' in this project so its own lockfile "
                    "pins the version this query compiles against",
                )
            )

    def _scope_of(
        self, package: Package, anchor: exp.Expr
    ) -> dict[str, _Function | WasmFunction]:
        """Every definition `package`'s lib files hold, read and parsed once per compile.

        Read on FIRST use of the package rather than up front: a package
        whose exports this query never calls into costs nothing, and a broken
        lib file only blocks the queries that reach for it. Reading also
        checks the manifest's promise -- each exported name defined in the
        file named for it -- and records which names are exported at all;
        the rest are the package's own.

        ``package.exports``' files and nothing else: a recipe's file is a
        query, and no path through this module opens one.
        """
        identity = (package.name, package.version)
        cached = self.scopes.get(identity)
        if cached is not None:
            return cached
        self._warn_about(package, anchor)
        scope, origin = _package_scope(package, anchor)
        _check_exported(package, scope, origin, anchor)
        self.scopes[identity] = scope
        self.exported[identity] = frozenset(package.exports)
        return scope

    # -- finding calls ----------------------------------------------------

    def _expand_statement(self, statement: exp.Expr, position: int) -> None:
        """Inline every call the statement writes, each into the query around it."""
        query = _query_node(statement)
        selects = [node for node in _preorder(statement) if isinstance(node, exp.Select)]
        for select in selects:
            self.site = self._site_of(query, select)
            self._expand_within(select, select, position, ())
        if isinstance(statement, exp.Copy) and selects:
            # A fan-out destination is written over the query's rows, so its
            # calls expand into that query.
            self.site = self._site_of(query, selects[0])
            for destination in statement.args.get("files") or []:
                if isinstance(destination, exp.Expr):
                    self._expand_within(destination, selects[0], position, ())
        self.site = None

    def _site_of(self, query: exp.Expr | None, select: exp.Select) -> _Site:
        """Where a call written in `select` puts the CTE it becomes."""
        if query is None:
            # A statement shaped like nothing the resolver knows; it is about
            # to be rejected, and the WITH goes somewhere it can be seen.
            return _Site(select, None)
        return _Site(query, _containing_cte(select))

    def _expand_within(
        self, root: exp.Expr, host: exp.Select, position: int, stack: tuple[str, ...]
    ) -> exp.Expr:
        """Inline the calls `root` writes, outermost first, into `host`.

        Returns what `root` became: a root that IS a call is replaced outright,
        and the scan continues over what took its place.
        """
        while True:
            site = self._next_call(root, position)
            if site is None:
                return root
            if isinstance(site, _WasmSite):
                replacement = self._adopt(site)
            elif site.row_source:
                self._expand_row_source(site, host, position, stack)
                continue
            else:
                replacement = self._expand_call(site, host, position, stack)
            site.node.replace(replacement)
            if site.node is root:
                root = replacement

    def _next_call(self, root: exp.Expr, position: int) -> _CallSite | _WasmSite | None:
        """The first call to a defined function in `root`'s own query, if any.

        A FROM item comes back as the ``exp.Table`` around the call, since that
        whole item is what a row source replaces, and a namespaced value call
        as the ``exp.Dot`` that qualifies it.
        """
        # A nested SELECT is its own query and gets its own pass, so the scan
        # stops at one -- but not at `root` itself, which is where it starts.
        stop = exp.Select if isinstance(root, exp.Select) else None
        for node in _preorder(root, stop=stop):
            if isinstance(node, exp.Table):
                site = self._row_source_site(node)
            elif isinstance(node, exp.Dot):
                site = self._qualified_site(node)
            elif isinstance(node, exp.Anonymous):
                site = self._bare_site(node)
            else:
                continue
            if site is None:
                continue
            if isinstance(site, _WasmSite):
                return site
            self._check_shape(site)
            self._check_defined(site.function, site.node, position)
            return site
        return None

    def _check_two_segment(
        self, segments: tuple[str, ...], call: exp.Anonymous, node: exp.Expr
    ) -> None:
        """A two-segment ``ffrwd.<name>`` that names an installed package, not a macro.

        Two segments under the official namespace is always a macro, so a
        package is never reached that way. Saying which package was meant
        beats leaving lower to report an unknown macro and name only macros.
        """
        packages = self.packages
        if packages is None or len(segments) != 1 or segments[0] != MACRO_NAMESPACE:
            return
        name = _call_name(call)
        found = packages.get(f"{MACRO_NAMESPACE}/{name}")
        if found is None:
            return
        raise _error(
            ErrorCode.UNKNOWN_FUNCTION,
            f"{MACRO_NAMESPACE}.{name}() is two segments, which is always a "
            f"macro, and there is no macro '{name}'",
            node,
            hint=f"package '{found.name}' is called in full: "
            f"{MACRO_NAMESPACE}.{name}.<member>(...)",
        )

    def _site(
        self, function: _Function | WasmFunction | None, call: exp.Anonymous, node: exp.Expr
    ) -> _CallSite | _WasmSite | None:
        """The site a resolved definition earns, by what kind of definition it is."""
        if function is None:
            return None
        if isinstance(function, WasmFunction):
            if isinstance(node, exp.Table):
                error = _not_a_table(function, node)
                if error is not None:
                    raise error
            return _WasmSite(function, call, node)
        return _CallSite(function, call, node)

    def _adopt(self, site: _WasmSite) -> exp.Expr:
        """A package's wasm declaration, adopted into the script, and the call rewritten.

        The declaration reaches lowering on ``Script.wasm`` like one the script
        wrote, under the call path it was written as -- a name no identifier
        can collide with, and the one a message about the call should show.
        One declaration per member however many calls reach it, and two
        versions of one package are two, told apart by version.
        """
        declared = site.declared
        key = (declared.package, declared.package_version, declared.name)
        name = self.adopted.get(key)
        if name is None:
            name = declared.called
            if name in self.wasm:
                name = f"{name}@{declared.package_version}"
            self.adopted[key] = name
            # Ahead of every statement: a library is already declared when the
            # query that calls it is written.
            self.wasm[name] = replace(
                declared, name=name, position=-1, package="", package_version=""
            )
        line, col = _pos(site.node)
        adopted = exp.Anonymous(this=name, expressions=list(site.call.expressions))
        adopted.meta.update({"line": line, "col": col, "start": 0, "end": 0})
        if isinstance(site.node, exp.Table):
            # A FROM-position call's `node` is the whole Table, alias
            # included (`_row_source_site`) -- replacing it with the bare
            # call would discard the alias the query bound it under, the
            # same fix `_expand_row_source` already makes for a RETURNS
            # TABLE function's FROM item.
            return exp.Table(this=adopted, alias=site.node.args.get("alias"))
        return adopted

    # -- wasm calls -------------------------------------------------------

    def _rewrite_sink_copy(self, statement: exp.Expr) -> None:
        """A sink function in TO position, rewritten into the query's SELECT list.

        ``COPY (SELECT a, b) TO snk(x)`` reads as the call ``snk(a, b, x)``:
        the SELECT list carries the streams -- and the annotation column,
        where one is written -- and the destination carries the values. The
        rewrite makes it exactly that call, so everything about a consumer
        call holds of a sink's without a second copy of the rules. The
        destination keeps the bare name, which is what marks the COPY as a
        module sink downstream.
        """
        if not isinstance(statement, exp.Copy):
            return
        files = statement.args.get("files") or []
        target = files[0] if len(files) == 1 else None
        if not isinstance(target, exp.Anonymous):
            return
        declared = self.wasm.get(_call_name(target))
        if declared is None:
            return  # not a wasm call; the resolver's own refusal stands
        if not declared.is_sink:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"COPY TO names '{declared.name}', which returns "
                f"{declared.written_returns}",
                target,
                hint="only a RETURNS sink function may be a COPY destination",
            )
        query = statement.this
        select = query.this if isinstance(query, exp.Subquery) else query
        if not isinstance(select, exp.Select):
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{declared.name}' is a sink destination over something "
                "other than one SELECT",
                target,
                hint=_WASM_SINK_HINT,
            )
        items: list[exp.Expr] = []
        for item in select.expressions:
            if isinstance(item, exp.Star):
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"'{declared.name}' is a sink destination, and its SELECT "
                    "list is a star",
                    item,
                    fallback=target,
                    hint="spell out the streams the sink reads: "
                    f"COPY (SELECT <stream>, ...) TO {declared.name}(...)",
                )
            items.append(item.this if isinstance(item, exp.Alias) else item)
        line, col = _pos(target)
        call = exp.Anonymous(
            this=declared.name,
            expressions=[*items, *[value.copy() for value in target.expressions]],
        )
        call.meta.update(
            {
                "line": line,
                "col": col,
                "start": 0,
                "end": 0,
                "sink_destination": True,
                # Where the SELECT's columns end and the TO call's values
                # begin. A sink reading an ARRAY of streams has no fixed
                # argument count, so this is what tells the two apart.
                SINK_STREAMS: len(items),
            }
        )
        select.set("expressions", [call])

    def _check_wasm_calls(self, statement: exp.Expr, position: int) -> None:
        """Check every call to a wasm function, and leave it where it is.

        There is no body to inline, so the call node reaches lowering intact.
        What is checked here is what a written call can be wrong about on its
        own: where it is written, when it is written, and its arguments
        against the signature.
        """
        for node in statement.walk():
            if isinstance(node, exp.Table):
                self._reject_wasm_row_source(node)
                continue
            if not isinstance(node, exp.Anonymous):
                continue
            if _qualified_by(node) or isinstance(node.parent, exp.Table):
                continue  # a qualifier owns the name under it
            if isinstance(node.parent, exp.Copy) and node.arg_key == "files":
                continue  # the rewritten SELECT call carries the checks
            declared = self.wasm.get(_call_name(node))
            if declared is None:
                continue
            if declared.is_sink and not node.meta.get("sink_destination"):
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"function '{declared.name}' returns sink, and this call "
                    "is not a COPY destination",
                    node,
                    hint=_WASM_SINK_HINT,
                )
            if declared.is_source:
                # A Table-position call never reaches here: its parent branch
                # above continues past it. Anything that does is a source
                # called as a stream function instead of a row source.
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"function '{declared.name}' returns source, and this "
                    "call is not in FROM",
                    node,
                    hint=_WASM_SOURCE_CALL_HINT,
                )
            self._check_wasm_position(declared, node, position)
            arguments = [
                argument for argument in node.expressions if isinstance(argument, exp.Expr)
            ]
            self._check_wasm_arguments(declared, node, arguments)
            self.wasm_used.add(declared.name)

    def _reject_wasm_row_source(self, item: exp.Table) -> None:
        """A wasm function in FROM: refused unless it is a source, which is exactly a table.

        A source is the only wasm return that belongs here, so a legal one is
        marked used like any other call -- this branch is the only place a
        Table-position wasm call is ever seen, so nothing else would.
        """
        call = item.this
        if not isinstance(call, exp.Anonymous):
            return
        declared = self.wasm.get(_call_name(call))
        if declared is None:
            return
        error = _not_a_table(declared, item)
        if error is not None:
            raise error
        self.wasm_used.add(declared.name)

    def _check_wasm_position(
        self, declared: WasmFunction, call: exp.Expr, position: int
    ) -> None:
        """A call may only name a declaration an earlier statement wrote."""
        if declared.position <= position:
            return
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"function '{declared.name}' is used before it is defined",
            call,
            hint="define every function before the statements that call it",
        )

    def _check_sink_arguments(
        self,
        declared: WasmFunction,
        call: exp.Anonymous,
        arguments: list[exp.Expr],
        streams: int,
    ) -> None:
        """A sink call's arguments, split where the rewrite says they split.

        The leading `streams` arguments came out of the SELECT list and are
        checked for being streams at all; which parameter each fills is a
        question of KIND, and kinds are settled in lowering. The rest are the
        module's own values, held to the same rules any call's are.
        """
        for argument in arguments[:streams]:
            written = _argument_kind(argument, self.wasm)
            if written is None or written == "stream":
                continue
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{declared.name}() is a sink destination, and its SELECT list "
                f"carries {_KIND_NAMES.get(written, written)}",
                argument,
                fallback=call,
                hint=f"a sink reads the streams its SELECT list names: "
                f"COPY (SELECT <stream>, ...) TO {declared.name}(<values>)",
            )
        values = arguments[streams:]
        positions = declared.value_params
        plural = "" if len(values) == 1 else "s"
        if len(values) > len(positions):
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{declared.name}() got {len(values)} value argument{plural}, "
                f"but it declares {len(positions)}",
                call,
                hint=declared.signature,
            )
        unfilled = next((p for p in positions[len(values) :] if p.default is None), None)
        if unfilled is not None:
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{declared.name}() got {len(values)} value argument{plural}, but "
                f"its parameter '{unfilled.name}' has no DEFAULT",
                call,
                hint=declared.signature,
            )
        for param, argument in zip(positions, values):
            if _call_name(argument) == _INPUT:
                raise _error(
                    ErrorCode.UDF_ARG_TYPE,
                    f"{declared.name}() cannot take input() as its '{param.name}' "
                    "argument: input() mints a FROM item, not a value",
                    argument,
                    fallback=call,
                    hint=_ARG_HINT,
                )
            written = _argument_kind(argument, self.wasm)
            if written is None or written == _declared_kind(param.type):
                continue
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{declared.name}() takes {param.type} as its '{param.name}' "
                f"argument, got {_KIND_NAMES.get(written, written)}",
                argument,
                fallback=call,
                hint=declared.signature,
            )

    def _check_wasm_arguments(
        self, declared: WasmFunction, call: exp.Anonymous, arguments: list[exp.Expr]
    ) -> None:
        """Arity and argument shapes, against the declared signature.

        The same rules a sql function's call carries, minus the ones about a
        body: the leading argument is the stream the module filters, and the
        rest are the values it is configured with. An annotation column
        usually takes no argument of its own -- the call that fills it is the
        leading one -- but a call may write one, and does when the rows are
        not the ones its stream argument arrived with.
        """
        streams = call.meta.get(SINK_STREAMS)
        if declared.is_sink and isinstance(streams, int):
            self._check_sink_arguments(declared, call, arguments, streams)
            return
        positions = declared.written_params
        if _writes_annotation(declared, arguments, self.wasm):
            positions = declared.annotation_written_params
        plural = "" if len(arguments) == 1 else "s"
        if len(arguments) > len(positions):
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{declared.name}() got {len(arguments)} argument{plural}, but it "
                f"declares {len(positions)}",
                call,
                hint=declared.signature,
            )
        unfilled = next(
            (p for p in positions[len(arguments) :] if p.default is None), None
        )
        if unfilled is not None:
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{declared.name}() got {len(arguments)} argument{plural}, but its "
                f"parameter '{unfilled.name}' has no DEFAULT",
                call,
                hint=declared.signature,
            )
        for param, argument in zip(positions, arguments):
            if _call_name(argument) == _INPUT:
                raise _error(
                    ErrorCode.UDF_ARG_TYPE,
                    f"{declared.name}() cannot take input() as its '{param.name}' "
                    "argument: input() mints a FROM item, not a value",
                    argument,
                    fallback=call,
                    hint=_ARG_HINT,
                )
            if param.annotation is not None:
                # An annotation column's shape is not a kind: what it has to
                # be is the record the producing module publishes, which
                # lowering matches against the module's own rows.
                continue
            written = _argument_kind(argument, self.wasm)
            if written is None or written == _declared_kind(param.type):
                continue
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{declared.name}() takes {param.type} as its '{param.name}' "
                f"argument, got {_KIND_NAMES.get(written, written)}",
                argument,
                fallback=call,
                hint=declared.signature,
            )

    def _bare_site(self, call: exp.Anonymous) -> _CallSite | _WasmSite | None:
        """An unqualified ``fn(...)`` value call, if something in scope defines `fn`."""
        # A qualifier owns the name under it: the Anonymous inside a Dot or a
        # FROM item was already offered to the branch that reads the qualifier,
        # and a definition may not shadow what `ffmpeg.` or `me.` resolves.
        if isinstance(call.parent, exp.Dot | exp.Table):
            return None
        return self._site(self._visible(_call_name(call)), call, call)

    def _qualified_site(self, node: exp.Dot) -> _CallSite | _WasmSite | None:
        """A qualified value call -- ``x.fn(...)`` or ``ns.pkg.fn(...)`` -- if it names one."""
        call = node.expression
        if not isinstance(call, exp.Anonymous):
            return None
        segments = _dot_segments(node.this)
        if segments is None or len(segments) > 2:
            return None
        if _dialect_owned(segments):  # a filter or a macro; lower resolves it
            self._check_two_segment(segments, call, node)
            return None
        return self._site(self._member(segments, call, node), call, node)

    def _row_source_site(self, item: exp.Table) -> _CallSite | _WasmSite | None:
        """A FROM-position call, bare or qualified, if something defines it."""
        call = item.this
        if not isinstance(call, exp.Anonymous):
            return None
        catalog, db = item.args.get("catalog"), item.args.get("db")
        if catalog is None and db is None:
            function = self._visible(_call_name(call))
        elif not isinstance(db, exp.Identifier) or (
            catalog is not None and not isinstance(catalog, exp.Identifier)
        ):
            return None  # something stranger than a plain qualified path
        else:
            segments = (
                (_ident_name(catalog), _ident_name(db))
                if isinstance(catalog, exp.Identifier)
                else (_ident_name(db),)
            )
            if _dialect_owned(segments):  # `ffmpeg.<source>()`; lower resolves it
                self._check_two_segment(segments, call, item)
                return None
            function = self._member(segments, call, item)
        return self._site(function, call, item)

    def _check_shape(self, site: _CallSite) -> None:
        """A call has to be written where what the function returns belongs."""
        function = site.function
        if site.row_source and not function.returns_rows:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"function '{function.qualified}' returns a value, not a table",
                site.node,
                hint="call it where its value belongs: a WHERE predicate, a "
                "tags field, a fan-out TO",
            )
        if not site.row_source and function.returns_rows:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"function '{function.qualified}' returns a table, not a value",
                site.node,
                hint=f"call it in FROM: FROM {function.qualified}(...) AS t, then read "
                f"{_listed(function.columns)} off the alias; a field read off the "
                "call itself reads it once per field, which would mint one input "
                "per read",
            )

    def _check_defined(self, function: _Function, node: exp.Expr, position: int) -> None:
        """A call may only name a function an earlier statement defined.

        A library's definitions were written before the script was, so the
        ordering rule is the script's own alone.
        """
        if function.library or function.position <= position:
            return
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"function '{function.name}' is used before it is defined",
            node,
            hint="define every function before the statements that call it",
        )

    # -- inlining one call ------------------------------------------------

    def _enter(self, function: _Function, call: exp.Expr, stack: tuple[str, ...]) -> None:
        """What every inlining costs: the cycle check, the budget, the used mark."""
        key = function.qualified
        if key in stack:
            chain = " -> ".join([*stack[stack.index(key) :], key])
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"function '{key}' is recursive: {chain}",
                call,
                hint="a filtergraph is acyclic; a function may not call itself, "
                "directly or through another",
            )
        self.budget -= 1
        if self.budget < 0:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"this script inlines more than {_EXPANSION_BUDGET} function calls",
                call,
                hint="flatten the nesting: fewer functions calling functions",
            )
        function.used = True

    def _arguments(
        self, function: _Function, call: exp.Anonymous, host: exp.Select, position: int
    ) -> list[exp.Expr]:
        """The written arguments, expanded and checked against the signature.

        An argument is the CALLER's text, so its own calls expand in the
        caller's context -- f(f(x)) is nesting, never recursion.
        """
        raw = [node for node in call.expressions if isinstance(node, exp.Expr)]
        arguments = [self._expand_within(node, host, position, ()) for node in raw]
        self._check_arguments(function, call, arguments)
        return arguments

    def _bound(self, function: _Function, arguments: list[exp.Expr]) -> list[exp.Expr]:
        """`arguments`, in signature order, with a default filled in for each
        parameter the caller left NULL or unwritten.

        NULL is absence throughout the dialect -- an unset variable
        substitutes to it -- so a NULL argument to a defaulted parameter
        takes the default the same way omitting it does; a caller who means
        NULL itself has no defaulted parameter to pass it to.
        """
        bound = list(arguments)
        for index, param in enumerate(function.params):
            if index < len(bound):
                if param.default is not None and _is_null(bound[index]):
                    bound[index] = copy.deepcopy(param.default)
                continue
            assert param.default is not None  # _check_arguments already enforced this
            bound.append(copy.deepcopy(param.default))
        return bound

    def _instance(
        self, site: _CallSite, arguments: list[exp.Expr]
    ) -> tuple[exp.Select, int]:
        """One private copy of the body: aliases renamed, positions stamped, arguments bound."""
        function = site.function
        body = copy.deepcopy(function.body)
        index = len(self.expansions)
        line, col = _pos(site.node)
        self.expansions.append(_Expansion(function.qualified, line, col))
        _rename(body, self._fresh_aliases(function, index))
        self._stamp(body, index)
        bound = self._bound(function, arguments)
        _substitute(body, {p.name: a for p, a in zip(function.params, bound)})
        return body, index

    def _expand_call(
        self, site: _CallSite, host: exp.Select, position: int, stack: tuple[str, ...]
    ) -> exp.Expr:
        """The expression the call stands for, with the body spliced into `host`."""
        function = site.function
        self._enter(function, site.node, stack)
        arguments = self._arguments(function, site.call, host, position)
        body, _ = self._instance(site, arguments)
        # The body is the package's or the script's text, so its own bare calls
        # resolve where it was written, not where it was called from.
        with self._scoped(function.identity):
            self._expand_within(body, host, position, (*stack, function.qualified))
        _splice(host, body)
        projection: exp.Expr = body.expressions[0]
        inner = projection.this if isinstance(projection, exp.Alias) else None
        return inner if isinstance(inner, exp.Expr) else projection

    def _expand_row_source(
        self, site: _CallSite, host: exp.Select, position: int, stack: tuple[str, ...]
    ) -> None:
        """Turn one FROM-position call into a generated CTE, and read that instead.

        The body becomes a relation of its own, which is what makes the call
        site carry the body's ROW COUNT: splicing its FROM items into the host
        would hand the host a product it never wrote.
        """
        item = site.node
        if not isinstance(item, exp.Table):  # unreachable: only a Table is a row source
            return
        function = site.function
        # `db` (two-part) and `catalog` (three-part) are a package call's own
        # qualifiers, and nothing else reaches here carrying one: an unclaimed
        # qualifier never resolves.
        _check_query_args(
            item, frozenset({"this", "alias", "db", "catalog"}), "a table function call"
        )
        self._enter(function, item, stack)
        arguments = self._arguments(function, site.call, host, position)
        body, index = self._instance(site, arguments)
        # The body is its own query now, so its own calls expand into it.
        with self._scoped(function.identity):
            self._expand_within(body, body, position, (*stack, function.qualified))
        _name_columns(body, function.columns or (), function.name, item)
        name = self._fresh_name(f"{function.name}_{index + 1}")
        self._add_cte(name, body)

        identifier = exp.Identifier(this=name, quoted=False)
        line, col = _pos(item)
        identifier.meta.update({"line": line, "col": col, "start": 0, "end": 0})
        alias = item.args.get("alias")
        if alias is None:
            # Unwritten, the alias is the function's own name, as Postgres has it.
            alias = exp.TableAlias(this=exp.Identifier(this=function.name, quoted=False))
        item.replace(exp.Table(this=identifier, alias=alias))

    def _fresh_name(self, base: str) -> str:
        """A name for a generated CTE that nothing in the script has claimed."""
        name = base
        while name in self.taken:
            name += "_"
        self.taken.add(name)
        return name

    def _add_cte(self, name: str, body: exp.Select) -> None:
        """Bind `body` as one more CTE of the query the call was written in."""
        site = self.site
        if site is None:  # unreachable: every expansion runs under one statement
            raise _error(ErrorCode.UNSUPPORTED_SQL, "a table function has no query to join", body)
        with_ = site.query.args.get("with_")
        if not isinstance(with_, exp.With):
            with_ = exp.With(expressions=[])
            site.query.set("with_", with_)
        entries = list(with_.expressions)
        at = len(entries)
        if site.before is not None:
            for index, entry in enumerate(entries):
                if entry is site.before:
                    at = index
                    break
        alias = exp.TableAlias(this=exp.Identifier(this=name, quoted=False))
        entries.insert(at, exp.CTE(this=body, alias=alias))
        with_.set("expressions", entries)

    def _fresh_aliases(self, function: _Function, index: int) -> dict[str, str]:
        """A name per body alias that nothing in the script has already claimed."""
        mapping: dict[str, str] = {}
        for alias in sorted(function.aliases):
            fresh = f"{function.name}_{index + 1}_{alias}"
            while fresh in self.taken:
                fresh += "_"
            self.taken.add(fresh)
            mapping[alias] = fresh
        return mapping

    def _check_arguments(
        self, function: _Function, call: exp.Anonymous, arguments: list[exp.Expr]
    ) -> None:
        """Arity and what each argument's shape says, against the signature.

        Fewer arguments than parameters is legal exactly when every parameter
        left unwritten has a DEFAULT -- omission is trailing-only, so a call
        can never leave a gap earlier than its shortest written prefix.
        """
        plural = "" if len(arguments) == 1 else "s"
        if len(arguments) > len(function.params):
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{function.qualified}() got {len(arguments)} argument{plural}, but it "
                f"declares {len(function.params)}",
                call,
                hint=function.signature,
            )
        unfilled = next(
            (p for p in function.params[len(arguments) :] if p.default is None), None
        )
        if unfilled is not None:
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{function.qualified}() got {len(arguments)} argument{plural}, but its "
                f"parameter '{unfilled.name}' has no DEFAULT",
                call,
                hint=function.signature,
            )
        for param, argument in zip(function.params, arguments):
            if _call_name(argument) == _INPUT:
                raise _error(
                    ErrorCode.UDF_ARG_TYPE,
                    f"{function.qualified}() cannot take input() as its '{param.name}' "
                    "argument: input() mints a FROM item, not a value",
                    argument,
                    fallback=call,
                    hint=_ARG_HINT,
                )
            written = _argument_kind(argument, self.wasm)
            if written is None or written == _declared_kind(param.type):
                continue
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{function.qualified}() takes {param.type} as its '{param.name}' "
                f"argument, got {_KIND_NAMES.get(written, written)}",
                argument,
                fallback=call,
                hint=function.signature,
            )

    # -- positions --------------------------------------------------------

    def _stamp(self, body: exp.Expr, index: int) -> None:
        """Move the body's own line numbers into this expansion's private range."""
        base = _BODY_LINE_BASE + index * _BODY_LINE_SPAN
        for node in body.walk():
            line = node.meta.get("line")
            if isinstance(line, int):
                node.meta["line"] = base + min(line, _BODY_LINE_SPAN - 1)

    def _source(self, line: int) -> tuple[_Expansion, int, int, int] | None:
        """The expansion `line` came from, its body line, and the call site.

        The call site of a nested expansion is itself body text, so the walk up
        continues until it reaches a line the script actually has.
        """
        index, body_line = divmod(line - _BODY_LINE_BASE, _BODY_LINE_SPAN)
        if not 0 <= index < len(self.expansions):
            return None
        expansion = self.expansions[index]
        site = expansion
        for _ in range(len(self.expansions)):
            if site.line < _BODY_LINE_BASE:
                return expansion, body_line, site.line, site.col
            outer = divmod(site.line - _BODY_LINE_BASE, _BODY_LINE_SPAN)[0]
            if not 0 <= outer < len(self.expansions):
                break
            site = self.expansions[outer]
        return expansion, body_line, 1, 1

    def translate(self, err: FfrwdError) -> FfrwdError:
        """A rejection that landed on body text, said at the call site."""
        if err.line is None or err.line < _BODY_LINE_BASE:
            return err
        found = self._source(err.line)
        if found is None:
            return FfrwdError(err.code, err.message, line=1, col=1, hint=err.hint)
        expansion, body_line, line, col = found
        return FfrwdError(
            err.code,
            f"in the body of {expansion.name}() at body line {body_line}: {err.message}",
            line=line,
            col=col,
            hint=err.hint,
        )

    def settle(self, script: exp.Expr) -> None:
        """Flatten every stamped position onto its call site.

        Resolve is the last pass that can tell body text apart, so after it
        succeeds the expansions are ordinary query nodes and must anchor
        somewhere the reader can see.
        """
        for node in script.walk():
            line = node.meta.get("line")
            if not isinstance(line, int) or line < _BODY_LINE_BASE:
                continue
            found = self._source(line)
            node.meta["line"] = 1 if found is None else found[2]
            node.meta["col"] = 1 if found is None else found[3]
            node.meta["start"] = 0
            node.meta["end"] = 0
