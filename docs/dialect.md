# The ffrwd dialect

The particular SQL dialect is combination of two popular dialects. 

- **The statement and call surface is Postgres's** - `COPY ... TO`,
`$$`-quoted functions, `:'var'` substitution, `name => value` binding,
`VARIADIC`, `DEFAULT`, `::` casts
- **The value and row model is
BigQuery's** - `STRUCT` literals, arrays of structs, `* EXCEPT` /
`* REPLACE`, `ARRAY(select)` with `SELECT AS STRUCT` because a media
file is an array of structured things and BigQuery is the built
around that shape.

ffrwd accepts the surface below and rejects the rest with typed,
line-anchored errors ([errors.md](errors.md)). This page is the
language's formal shape: what exists, and what
does not.

## Statements

A query is ONE statement, or a script:

```
query   := select | copy
script  := (function ;)* (CREATE VIEW name AS select ;)* (copy ;)* copy?
function := CREATE FUNCTION name(param type [DEFAULT literal], ...) RETURNS rtype
            AS $$ select $$ LANGUAGE sql
          | CREATE FUNCTION name(stream wstype, ...,
                                 [ann annotation [DEFAULT NULL],]
                                 param type [DEFAULT literal], ...)
            RETURNS wrtype AS 'module', 'export' LANGUAGE wasm
          | CREATE FUNCTION name(param vtype, ...) RETURNS vtype
            AS 'module', 'export' LANGUAGE wasm
          | CREATE FUNCTION name(rows annotation) RETURNS annotation
            AS 'module', 'export' LANGUAGE wasm
rtype   := text | number | boolean | vector | <kind>_stream | chapter | cue
         | embedding
         | attachment | any of those with [] | TABLE(col type, ...)
wstype  := video_stream | audio_stream | either of those with []
wrtype  := wstype | sink | STRUCT(name wstype, name annotation)
annotation := STRUCT(field vtype, ...)[] | cue[]
vtype   := text | number | boolean | vector
select  := [WITH cte (, cte)*] SELECT columns FROM from [WHERE pred]
           [GROUP BY exprs] [ORDER BY exprs] [LIMIT count] [OFFSET count]
           (UNION ALL select)*
copy    := COPY ( select ) TO dest [WITH ( option value (, ...)* )]
cte     := name AS ( select )
dest    := 'path' | STDOUT | ( value-expression ) | sink(value, ...)
```

- A bare `SELECT` is a **table query**: the result prints (psql-style
  table, or CSV via `COPY ... TO STDOUT WITH (format 'csv')`), and
  ffmpeg never runs.
- A `COPY` with a media destination compiles to the ffmpeg command(s).
- A script's views compile into ONE ffmpeg invocation, one output per
  COPY.
- A **FUNCTION** is a reusable expression, inlined at compile time -
  it is the query you could have typed by hand. It must be defined
  before it is used and before the first `COPY`, every definition must
  be called, and a value-returning one is legal anywhere a value of its
  type is while a `TABLE`-returning one is a `FROM` row source only. A
  parameter may declare `DEFAULT literal`; calls are positional, so an
  omitted trailing argument takes it. Recipes
  [67-68](examples.md#67-write-a-function-and-reuse-it),
  [79](examples.md#79-give-a-parameter-a-default).
- A **`LANGUAGE wasm` function** names a wasm module and one export
  in it, and is called like any other function. It has no body to
  inline: the module runs in the `ffrwd-wasm` sidecar, so a query
  reaching one compiles to several processes joined by pipes rather
  than to one ffmpeg command, and `run` executes them together. Recipe
  [87](examples.md#87-run-a-wasm-module-over-the-picture).
- **`--jobs N`**, on `compile` and `run`, caps the sidecar's worker
  threads at N. The sidecar runs a pool sized to the machine's cores by
  default, and a module that describes itself as pure spreads across it
  with no flag at all; one that carries state between calls, and one
  reading encoded packets, run one call at a time whatever N is.
  `--jobs 1` hosts everything serially. The output is byte-identical at
  any N.
- A **sink `LANGUAGE wasm` function** (`RETURNS sink`) is a COPY
  destination: `COPY (SELECT <cells>) TO name(<values>)`. It declares
  value parameters only; the SELECT list supplies the ROWS it reads,
  each row a video cell, an audio cell, either NULL - the relation a
  manifest destination reads, so one row is a file-shaped sink and N
  rows are a ladder, every row one rendition. A sink that still
  declares stream parameters (`video_stream[]`, `audio_stream`) reads
  the SELECT's streams into them as before. How many streams of
  each kind a module reads is its own declaration, and a call whose
  count disagrees is refused. A sink writes nothing back: the module's
  own effects - an HTTP POST, a live broadcast, a line on stderr - are
  the output, so such a COPY names no file, and the run ends when the
  input drains. A sink reading decoded frames takes no `WITH` options;
  its value arguments configure it. One reading encoded packets sits
  behind an encoder the compiler places, whatever feeds it, and the
  COPY's `WITH` options shape that encoder the way they shape a
  file's - a value read once per row shapes each stream separately. A
  sink call anywhere else is refused. A
  module importing `wasi:http` or `wasi:sockets` runs only under the
  sidecar's matching per-module grant (`-http <module>`,
  `-net <module>`), which the compiler emits from the module's own
  describe; both are denied without it. Secrets do not belong in the
  value arguments - the query text is the command line. Recipes
  [98](examples.md#98-post-what-a-module-found-as-it-is-found),
  [99](examples.md#99-watch-the-frames-go-by).
- A **value-returning `LANGUAGE wasm` function** (`RETURNS text`,
  `number`, `boolean` or `vector`) takes no stream at all: every
  parameter is one of those same `vtype`s, matched name-for-name against
  the schema the module's own function declares, and RETURNS against
  what that function returns. It runs once per call, at compile time -
  like ffprobe, not like a filter. Recipe
  [89](examples.md#89-compute-a-tag-with-a-wasm-module). Over a row
  column it runs once per row, memoized on its arguments - the same
  per-row footing the built-in text and number functions have
  ([recipe 112](examples.md#112-a-function-over-a-caption-files-cues)).
  A `vector` is a JSON array of numbers - the wire shape of an
  embedding - and joins `vtype` for exactly this: a value function's
  own parameters and RETURNS, and an annotation field. It is not a
  scalar column type otherwise: it prints (capped, as a cell) but
  cannot be compared, concatenated, cast to text, or written as a tag.
  `cos_similarity(vector, vector) -> number` and `vector_length(vector)
  -> number` are the two built-ins over it, evaluated at compile time
  and, over a row column, once per row like any other. Recipe
  [114](examples.md#114-rank-rows-by-a-vector).
- A **rows `LANGUAGE wasm` function** (one annotation parameter,
  `RETURNS` an annotation) reads rows and writes rows with no stream
  at all. Its argument is the annotation column a run-time module
  produced - `fauxlate(captions(f.video[1]).cues)` - and its result
  is a row column of the declared type, standing wherever the
  producer's did: projected, it is a subtitle track; at a `.ndjson`
  destination it is the rows. It runs in the producer's own sidecar
  process, fed by the rows as they are written; the declared
  parameter is checked against what the module reads and the return
  against what it writes, and the producer's record has to be the one
  the function reads. Rows that exist before the run - a file's cues -
  are not its business: rewrite those one row at a time with the value
  form. Recipe
  [113](examples.md#113-translate-captions-as-they-are-produced).
- **`cue[]`** is shorthand for the cue record's own shape,
  `STRUCT(text text, start_t number, end_t number)[]`, wherever an
  annotation column is declared - a `RETURNS STRUCT`'s second field, or
  the matching parameter of a module that consumes them. It is the same
  record spelled out, and matches the same row schemas.
- **Narrowing the rows at run time** - `ARRAY(SELECT r FROM
  unnest(detect(v).boxes) r WHERE r.class = 'person')` - is the same
  annotation array with a predicate on it, and stands wherever the
  plain projection does: a consumer's annotation argument, a minted
  subtitle track, a `.ndjson` file. Nothing counts those rows at
  compile time, so unlike every other `ARRAY(SELECT ...)` this one is
  not evaluated: it compiles to one more node in the module network,
  fed by the call that produced the rows.
  The subquery selects the whole row (`SELECT r`) and carries a `FROM`
  and a `WHERE` and nothing else. The predicate holds `=`, `<>`, `<`,
  `<=`, `>`, `>=`, `AND`, `OR`, `NOT` and parentheses over the row's
  own fields and literals of their declared types. A field the record
  does not name, a literal of the wrong type for the field it is
  compared against, a reference past the alias, a computed projection,
  and anything else in the `WHERE` are each rejected where they are
  written.
- Trailing `;` allowed; `--` and `/* */` comments allowed. Unquoted
  identifiers fold to lowercase. View, CTE, and alias names share one
  flat namespace across the whole script.

## Projects and packages

A directory holding a `ffrwd.json` is a project, and the project is a
package. A package is named `<namespace>/<package>`, and that name is
the path a call writes: `imbcmdth/audio` is called as `imbcmdth.audio`.

```json
{ "name": "imbcmdth/audio", "version": "1.0.0",
  "description": "Volume, loudness, ducking",
  "bin": { "volume": "queries/volume.sql", "loudnorm": "queries/loudnorm-all.sql" },
  "lib": "src/audio.sql",
  "dependencies": { "tracks": "broadcast/tracks@^1.2.0" } }
```

`name` and `version` are required; the rest is optional. Each half of
the name is a lowercase plain identifier, and the first half - the
namespace - may not be `ffmpeg` or `wasm`. `ffrwd` is the official
namespace and packages do claim it.

`lib` and `bin` each take a string or a map. A
string names one file, its member named for the package segment
(`audio` in `imbcmdth/audio`) - `imbcmdth/deband`'s `"lib": "src/deband.sql"`
calls as `imbcmdth.deband(v)`. A map names several members, one file
per key, and there is no root-callable one - `imbcmdth/audio`'s `bin`
above reaches `imbcmdth.audio.volume`, never `imbcmdth.audio(...)`.
`lib` is the exports: each map key an exported function name, its
value the file defining it. `bin` is the recipes: a map key is a
command word (`[a-z][a-z0-9_-]*`), its value one query file. Several
keys may name one file; a file may define more than the manifest
exports, and the rest are private to the package. Each named member
must be defined in the file named for it. Every file value is one path
relative to the manifest, stays under it, is not a pattern, and must
exist. A manifest declaring neither `lib` nor `bin` is a consumer
project that holds dependencies, or a package that ships files rather
than SQL - `ffrwd/wasm` is one. `bins` and `libs` are gone; a
manifest that still writes either is rejected, its hint naming the
singular that replaced it.

The two halves are read by role. A lib file holds `CREATE FUNCTION`
definitions and nothing else - it is a library. A recipe's file is a whole query,
like any script

`dependencies` keys are package names - `<namespace>/<package>` - and
the values are the version range for each, recorded and shown, never
solved.

Eight more keys exist that no compile reads: seven the registry
stores, and `test`, which only `ffrwd publish` runs. The whole
manifest's shape is pinned by
[ffrwd-json.schema.json](ffrwd-json.schema.json).

```json
{ "keywords": ["depth", "nn"],
  "license": "MIT OR Apache-2.0",
  "homepage": "https://example.com/depth",
  "capabilities": ["nn"],
  "ffrwd": ">=0.9",
  "private": false,
  "test": "uv run pytest tests -q && cargo test --release",
  "models": { "depth": { "repo": "depth-anything/small", "revision": "v1",
                         "file": "onnx/model.onnx", "sha256": "<64 hex>" } } }
```

`keywords` is a list of at most 16 short labels, each at most 32
characters; the registry indexes and ranks over them. `ffrwd` is the
range of ffrwd versions the package declares it runs on, at most 64
characters, checked for shape and nothing more - recorded and shown,
never solved, and never a reason to refuse an install.

`license` is what the package is published under: one line of printable
text, at most 64 characters. Shape is all that is checked - an SPDX
identifier (`MIT`) and an expression over them (`MIT OR Apache-2.0`)
are the same thing here, and no list of licenses is consulted. Like the
description, it belongs to the version: changing it means publishing a
new one. A publish with no `license` warns and continues.

`homepage` is a project page for the package, shown on its registry
listing: an `http://` or `https://` URL, at most 300 characters,
trimmed - empty after trimming is the same as omitting the key.

`capabilities` is what the host must grant this package's modules:
`nn` to run a model, `http` to make HTTP requests, `udp` to open UDP
sockets. Anything else is refused, naming the list. These are grants, not wasi imports - clocks
and random are ambient, and a module having them declares nothing.
Absent means the same as empty. `publish` derives the set from what
each module says it does and refuses any disagreement in either
direction: a module needing a capability the manifest omits, and a
capability the manifest declares that no module needs.

`FFRWD_NET_POLICY=public` in the environment restricts both network
grants to public destinations: connects and sends to private
(RFC 1918), loopback, link-local, carrier-NAT, multicast and broadcast
addresses are refused, and an HTTP request's hostname is judged by the
addresses it resolves to, not by its spelling. Local binds are
untouched. Unset leaves the network unrestricted - local test relays
on loopback keep working. Any other value is an error at startup. The
hosted runner sets it; a module never resolves or reaches anything
internal to the machine it runs on.

`private` is a boolean, false when absent: true publishes each version
private -- installable with a token from the namespace, off the public
index. A version's visibility is stamped when it is published and never
changes, so flipping the key later only changes what the NEXT publish
is stamped with; every already-public version stays installable exactly
as it was. One rule follows: a version published public must have every
dependency resolvable among public versions.

`test` is one line of printable text, at most 500 characters: the
command `ffrwd publish` runs before it packs anything, in the package
directory, through the platform's own shell - so `&&` and a pipe mean
what you wrote. Exit 0 and the publish goes on; anything else stops it,
naming the command and its status, and its output is the terminal's
rather than something the publish captures. What the command checks is
the author's business - a recipe compiled with its own example values,
a test suite, a linter, a `cargo test`, all four. Absent, or empty, and
nothing is checked; nothing is compiled in its place either. It runs at
publish and nowhere else: the key rides in the archive because the
manifest does, `install` never reads it, and nothing there ever runs
it.

`models` pins the model each exported wasm function loads, keyed by the
EXPORT name. `repo` is `<owner>/<name>` on Hugging Face, `revision` one
branch, tag or commit, `file` one path inside that revision, and
`sha256` what its bytes must hash to. `install` fetches it into the
store beside the module's wasm file, named `<export>.onnx`, which is
where the compiler already looks for it (see
[Installing](#installing)). A model is therefore not in the archive: a
package that runs one is still kilobytes.

A model of several files is written as a LIST of pins, and the entries
may name different repositories:

```json
{ "models": { "transcribe": [
    { "repo": "<owner>/<name>", "revision": "<rev>",
      "file": "whisper-medium_beamsearch.onnx", "sha256": "<64 hex>" },
    { "repo": "<owner>/<name>", "revision": "<rev>",
      "file": "whisper-medium_beamsearch.onnx.data", "sha256": "<64 hex>" } ] } }
```

The FIRST entry is the graph, and lands as `<export>.onnx` like any
other. Every later entry lands beside it under its own file NAME -
`whisper-medium_beamsearch.onnx.data` above - because that is the name
the graph refers to it by. A later entry whose name would be
`<export>.onnx`, one whose path ends in no plain filename, one whose
name another entry already lands under, and an empty list are each
refused. `publish` asks the hub what every pinned file weighs and
records it beside the pin, so a package's page can say what installing
will download; a size that does not arrive is left out and the publish
carries on.

A query calls into an installed package one of two ways, always
written in full:

- **Three segments**, `<namespace>.<package>.<member>(...)` -
  `imbcmdth.audio.quieter(f.audio[1], 0.5)` - reaching any export
  whatever a project's own `dependencies` say.
- **Two segments**, `<namespace>.<package>(...)` - `imbcmdth.deband(v)` -
  the export a string `lib` names. A package whose `lib` is a map has
  none: the call is rejected, naming the exports to call by their own
  three-segment path instead.

There is no alias, so a ONE-segment qualifier is never a package
lookup at all: it is either a call into the query's own definitions
(bare, no qualifier) or, qualified, unresolvable - `UNKNOWN_FUNCTION`,
its hint naming the three-segment form calls across packages are
written as. `ffmpeg.<filter>` stays two-part and reserved, and is
never read as a package call.

`ffrwd` is the one namespace where both rules meet, and segment count
decides. Two segments, `ffrwd.<name>(...)`, is always a macro, whatever
is installed - a macro added to the language later can never be
shadowed by a package. Three, `ffrwd.<package>.<member>(...)`, is
always a package. Nothing is unreachable either way, because a package
named for a macro is refused when the manifest is read. A two-segment
call that names an installed `ffrwd/<package>` says so, and names the
three-segment form to write instead.

The VERSION a call reaches is whichever the CALLING package's own
manifest depends on, not a single project-wide binding: a call written
in the query itself resolves against the project's own `dependencies`,
a call written inside an installed package's lib file or recipe
resolves against THAT package's own. Two packages may each depend on a
different version of a third, and each keeps resolving against its
own - `install` walking `dependencies` records, on every package's own
lockfile entry, the exact version it resolved each of ITS dependencies
to (see [Installed packages](#installed-packages)).

Called as a value (`imbcmdth.audio.quieter(...)`) or as a row source
(`FROM imbcmdth.audio.pick('a.mka') t`), the call is expanded exactly
as a definition written into the query would be - same hygiene, same
arity and type checks, same command out. Nothing is prepended to the
script, and a package's names never enter the script's flat namespace.
A project's own definitions stay bare - `normalize(...)` inside the
project that defines it; unqualified names are never a package lookup.

`ffrwd list` prints what the project at the working directory and its
dependencies provide: the packages with their layer, the exports with
their signatures and files (the list is the manifest's; only parameter
types are read from the files), the recipes with the variables each
declares (read from its `-- variables:` header), and the dependencies.
`--json` for scripting.

The project is found by walking up from the query file's directory, or
from the working directory for a query typed on the command line; there
is no flag. Outside a project nothing changes: a namespaced call is
rejected as it always was. Library callers pass
`compile_sql(text, packages=ffrwd.discover(path))` rather than
relying on a working directory, and the MCP tools take the same path as
a `project` argument.

### Starting one

`ffrwd init` writes `ffrwd.json`, an empty `ffrwd.lock` and a
starter recipe into the working directory. The package segment is the
directory's name, folded to an identifier, unless `--name` says
otherwise; the namespace is `--namespace`'s, or derived from the git
remote `origin`'s owner, or required - `init` says which it used.
`--name <namespace>/<package>` gives the whole name at once. A name
nothing usable comes out of is rejected rather than guessed at. It
overwrites no file it would write, and what it writes reads back
through the same validation every other command applies.

The starter is a recipe, `recipes/resize.sql` declared as a map `bin`
entry, not an export: a lib must name a file defining its export, so a
fresh directory has nothing to declare one with.

`--rust` writes a wasm module package instead of the bare one. On top
of the manifest and the lockfile: `Cargo.toml` for a `cdylib` crate
named for the package segment, `build.rs`, `src/lib.rs` holding an
`invert` module, `src/invert.sql` declaring it as an export,
`recipes/invert.sql` calling that export, `.ffrwdignore`, `.gitignore`
and a `README.md`. The manifest depends on `ffrwd/wasm` at the current
world, and declares `capabilities` and `keywords` empty for their
author to fill in. `cargo build --target wasm32-wasip2 --release` then
`ffrwd publish` is the whole path from there.

`build.rs` puts the wit where `wit_bindgen::generate!({path: "wit"})`
reads it, from whichever source is available: `FFRWD_WIT_DIR` when the
environment names one, otherwise `ffrwd path ffrwd/wasm`. The crate's
`wit/` is build output and is gitignored.

### Running a recipe

`ffrwd run split-chapters -v source=film.mkv -v dest=out.mkv` runs a
recipe a package ships. `compile`, `explain` and `validate` take a
recipe name in the same position, so a recipe can be inspected as well
as run.

Which the positional is, in order:

1. Text beginning with `SELECT`, `COPY`, `CREATE` or `WITH` (past
   leading whitespace and comments) is SQL, always.
2. Otherwise, a name a discovered package ships a recipe under is that
   recipe: its file's text is the query.
3. Otherwise it is SQL, and fails as any other query text would.

A manifest declaring a recipe named for one of those four words is
rejected where it is written: rule 1 would never let it be reached.
A recipe name may also be qualified: `<namespace>/<package>:<name>`
names one entry of a map `bin`, `<namespace>/<package>` a string
`bin`'s recipe - the package spelled exactly as `install` takes it.
Either says which package when a bare name matches recipes in more
than one; a bare name that does is rejected naming each
`<namespace>/<package>:<name>` it could mean. Variables still come
from `-v name=value`; an unset one substitutes to `NULL` (see
[Variables](#variables)), and a rejection at
its point of use names what the recipe's `-- variables:` header
declares.

Dots stay SQL's: `ffrwd.examples.abr_ladder()` in a query is
unchanged, and the dotted form on the command line
(`ffrwd run ffrwd.examples.abr-ladder`) is refused with the
`/`-and-`:` spelling named. The two halves of a package name are SQL
identifiers - lowercase, digits, `_`, no hyphens - because queries
qualify calls with them; a recipe name is a command word, never
written in SQL, and may carry hyphens (`split-chapters`).

### Installed packages

`ffrwd.lock`, beside the manifest, records what the project
installed. It is machine-owned: installing writes it, nothing else
should. Each entry is one of two kinds, one per package VERSION - a
name may have more than one entry, since installing never removes one
version to make room for another.

```json
{ "format_version": 3,
  "reproducible": true,
  "dependencies": { "broadcast/tracks": "1.2.0" },
  "packages": [
    { "kind": "registry", "name": "broadcast/tracks", "version": "1.2.0",
      "sha256": "<64 hex>", "store": "v1/ab/<64 hex>",
      "dependencies": { "imbcmdth/audio": "2.1.0" } }
  ] }
```

A **registry** entry is keyed by the package's name and version, and
pins the sha256 of the ARCHIVE the package travels as - one gzipped
tar, built so the same content always produces the same bytes.
`install` hashes the bytes it downloaded before opening them: a
download that does not match the pin is discarded unopened and nothing
is written. What matches is extracted into the store under
`~/.cache/ffrwd/packages/`, and the extractor takes regular files and
directories under the package root and nothing else - no absolute
paths, no `..`, no links, no devices, and a member count and
uncompressed size cap. Reading a stored package hashes nothing; content
that is missing from the store is a rejection naming the package, never
a fall back to what is there. Its own `dependencies` is what ITS
manifest's `dependencies` resolved to when this entry was installed -
package name to the exact version, never a range - which is what lets
a call written inside that package resolve at the right version even
when another installed package depends on a different one of the same
name.

Two entries pinning one package at one version are rejected; two
different versions are not - that is the whole point of carrying
several.

The lockfile's own top-level `dependencies` is the same shape, one
level up: what THIS project directly installed, package name to the
exact version - what a call written in the project's own query
resolves against. It is separate from the manifest's `dependencies`,
which records the range as written and is never solved.

**Links** live in `ffrwd.links` beside the lockfile, machine-local and
not for version control:

```json
{ "format_version": 1,
  "machine_local": "the directories this machine reads packages out of, live; not for version control",
  "links": [ { "path": "../my-lib" } ] }
```

A link names a directory and nothing else - the package's name comes
from the manifest there, so renaming the package needs no re-link. Its
`ffrwd.json` is read like any other manifest, so an edit lands in the
next compile - which is the point, and which no digest could survive.
A lockfile written by an older ffrwd may still hold `"kind": "link"`
entries; they are read, and the next `link`, `unlink` or `install`
moves them into `ffrwd.links` and out of the lockfile.

A package resolves through three layers, the first claim on its name
winning: the project's own manifest, then its links file and lockfile,
then the machine-wide pair a global install and a bare `ffrwd link`
write. A link claims its name over anything installed under it. Two of those
layers are worth saying out loud, so a compile reports them without
refusing: resolving inside a project but landing on the machine-wide
layer, and compiling against a link. Each is reported once per package
- as a `warning:` line on stderr from the CLI, in the `warnings` array
of an MCP tool result, and through `compile_sql`'s optional
`on_warning` callback for a library caller.

A third travels the same channel and is not about packages at all. A
stream array a file has no tracks for - `f.subtitle` where the file
carries none - is empty, contributes no streams, and warns
(`EMPTY_STREAM_ARRAY`). That is what `unnest` of it already does and
what `SELECT *` already does; naming the column is the third spelling
of the same thing. A sink left with NO streams at all is still a
rejection, since it would write a file with nothing in it.

### The registry

Two hosts answer, and they are two settings. `FFRWD_REGISTRY` is where
the detail documents are served from: `p/<namespace>/<package>.json` is
one package's detail - every published version, and per version the
archive's sha256, size and what it provides. `FFRWD_API` is where the
three things a served file cannot do live: the search function, the
endpoint that signs an archive URL, and the one that serves a private
package's detail.

There is no catalogue file and no index step. A package is resolved by
fetching its own detail document; the registry not having it is what a
missing document means, and the suggestion in that rejection comes from
the search function - best effort, so a search that cannot be reached
costs the suggestion and nothing else.

`FFRWD_REGISTRY` may also be a `file://` URL or a plain directory path
holding `p/` and `archives/<sha256>`. That is the offline registry: no
signing, no search function, the files read straight off the
filesystem. Search there reads the detail documents themselves and
filters case-insensitively over each package's name, description and
exported function names.

Over HTTP, EVERY archive is fetched through a signed URL. The client
asks the archive endpoint for one, sending its token when it has one,
and then makes a plain GET on the URL it is handed. A public version's
archive signs without a token; a private one needs it, and a refusal
with no token on this machine says to run `ffrwd login`. That request
is also where an install is counted, which is why it has no bypass.

A package the public documents do not list may still be one this
account may see: with a token saved, a missing public document is
retried against the private detail endpoint before the package is
called missing.

### Searching

`ffrwd search tracks` asks the registry what it ranks for the term and
prints it, most relevant first: an exact name, then a near name, then
the text of the description, the keywords and the names a package
exports. The table shows each package's installs over the trailing
week. No term browses everything, a term matching nothing is an empty
table and exit 0, and `--json` emits the same results for scripting.

### Installing

`ffrwd install broadcast/tracks` fetches the package's detail
document, asks the registry to sign the archive that document names,
verifies the bytes against the sha256 recorded there before opening
them, writes the content into the store, fetches every model the
manifest pins, and then pins the package - a registry entry in the
lockfile and a dependency in `ffrwd.json`. Nothing is recorded before
all of that is on disk, so an install that fails leaves the project
pinning only what it had.

A model comes from Hugging Face rather than from the registry, wherever
the registry itself is: `https://huggingface.co/<repo>/resolve/<revision>/<file>`,
hashed against the manifest's pin while it streams and moved into place
only once it matches. It lands beside the module's wasm file as
`<export>.onnx`. One already there and matching is not fetched again;
one that will not verify fails the install, naming the model and its
pin.

`install <pkg>@<version>` takes that version; without one, the highest
published version is taken and written exact. Exact pins only: a
manifest `dependencies` range is recorded and shown, never solved.

It then walks the installed package's own manifest `dependencies` and
installs each of THOSE the same way, recursively - each at its highest
published version, exactly as a direct install resolves one. A
dependency already pinned in the lockfile at the exact version wanted
is left alone: not refetched, not walked again. A different version of
the same name already pinned is not a conflict - install never
arbitrates between versions of one package, so it simply pins the new
one beside the old, and each installed package keeps resolving its own
calls against the version IT depends on. A cycle in that walk is
rejected, naming the loop. Only the package named on the command line
is recorded in the project's manifest; what came along transitively is
the lockfile's business, reported as what was brought along.

A version already in the store is not downloaded again - the same
content, installed into a second project or globally, is stored once.

The dependency is recorded in the manifest keyed by the package's own
name - `install broadcast/tracks` writes
`"broadcast/tracks": "1.2.0"`. A global install (`-g`) has no
manifest, so nothing is recorded there, but its lockfile's own
top-level `dependencies` still records what was directly asked for.
Installing a package already directly pinned at another version
changes what the project points at; the old entry is not removed, in
case something else installed still depends on it.

`-g` and the project rule are `link`'s, below.

`ffrwd install` with no package installs the project standing here:
every dependency its own manifest pins, at the written version, plus
its pinned models and the runtime its modules load - everything
installing this package from the registry would have fetched. A fresh
clone of a package's repository builds and publishes after one bare
install. `-g` without a package is an error; machine-wide installs
name what to fetch.

### Where a package is

`ffrwd path broadcast/tracks` prints the directory an installed
package is read out of: the store directory for an installed one, the
linked directory for a link. It resolves through the lockfile
`install` writes and the links file beside it, or the machine-wide
pair with `-g`, and prints one line - the path and no decoration,
because a build script reads it. A package neither file records is a
rejection naming `ffrwd install`.

`ffrwd/wasm` is the package that makes it worth having. It ships no
exports and no recipes, only `wit/av.wit`, and its VERSION IS the
world version: `ffrwd/wasm@0.9.0` carries the `ffrwd:av@0.9.0` world,
immutable and resolvable forever like any other version. A module
depends on a wit the way it depends on anything else - a manifest
`dependencies` entry, a lockfile pin - and its `build.rs` asks
`ffrwd path ffrwd/wasm` where the file landed.

From `ffrwd:av@0.11.0` a windowed module's `process` borrows its
window instead of receiving it as a list: `len`, `pts` and `rows` are
cheap, and `fetch` copies one payload's bytes in on demand, so a
module reading one frame of a wide window never pays for the rest.
Modules built against earlier worlds load and run unchanged.

### Linking

Linking is npm's two commands, each with its effects in one place.
`ffrwd link`, bare, run in the package's directory: installs what its
manifest pins - its own lockfile, the shared store - and records
name -> directory in the machine-wide `ffrwd.links` under the cache
directory. `ffrwd link <namespace>/<package>`, run in a consuming
project: records the name in the project's `ffrwd.links` beside its
lockfile, which itself is untouched. A name nothing on this machine
links is refused, hint naming the first command; so is a directory
path - the consumer names packages, never directories.

The name resolves through the machine-wide record, so re-running the
bare form from a new directory re-points every consumer at once. The
linked package resolves its own calls through its OWN lockfile against
the shared store; a tree whose manifest drifts past that lockfile
after linking is refused at the next compile, hint naming the re-link.
The consuming project's own queries never see the linked package's
dependencies. Linking a package something else already pins leaves the
pin in place; the link answers while it stands.

`ffrwd unlink`, bare in the package's directory, removes the
machine-wide record - consumers refuse at their next compile, naming
the way back. `ffrwd unlink <name>` in a project removes that
project's record (a directory works too, for an old record whose
manifest is gone). `ffrwd unlink -g <name>` removes the machine-wide
record from anywhere, which is how one whose directory is gone gets
cleaned up; `link -g` is retired, the bare form being machine-wide
already.

A record an older ffrwd wrote - a directory path in a project's
`ffrwd.links`, or a link entry in the lockfile itself - still
resolves. The next `link`, `unlink` or `install` migrates it into
`ffrwd.links`: by name when the machine-wide file links its package, a
path otherwise. Naming a package to link outside a project is a usage
error (exit 2) pointing at `init`; no command creates a lockfile
anywhere but the linked package's own directory.

Every write to these files replaces it in one step and pins LF endings,
and each is written in insertion order with no timestamp, so writing
the same set of packages twice produces the same bytes. Writing the
last link away removes `ffrwd.links` itself.

### Publishing

`ffrwd login --token <token>` saves the token this machine publishes
with, in `%APPDATA%\ffrwd\credentials.json` on Windows and
`$XDG_CONFIG_HOME/ffrwd/credentials.json` (or `~/.config/ffrwd/`)
elsewhere, readable by its owner and nobody else. Not under
`~/.cache/`: a cleared cache must not log anyone out. `FFRWD_TOKEN`
overrides the file, which is what CI uses. A token is `ffrwd_` and 32
bytes of base64url; `login` checks that shape and reaches nothing -
whether the token is live is the registry's answer at the first command
that uses it. `ffrwd logout` removes the file.

`ffrwd publish` publishes the package at or above the working
directory. The publisher has the whole toolchain, so the validation is
local and it is the whole of it, run before anything leaves the
machine:

- the manifest reads, and its name is one a package may have - which
  includes the refusal of a package named for a macro, since the macro
  list is the compiler's;
- every export parses and defines what the manifest says it does;
- every dependency the manifest names resolves in the registry;
- the sidecar describes every module, which is where the version's
  capabilities come from - a module that runs a model makes the
  version an `nn` one;
- every model pinned names an export one of those modules declares;
- the manifest's own `test` command runs, and its exit status decides
  whether the publish goes on. A manifest that declares none is not
  checked: what a package promises about itself is the package's to
  say, and a version nobody vouched for is one anybody may publish.

Then the directory is packed - the same deterministic archive
`install` verifies - and one request carries the bytes and the JSON the
registry stores: the version's detail document, the recipe sources, the
capabilities, and the manifest's `ffrwd`, `keywords`, `license` and
`models`, each pin carrying what the hub says the file weighs.

What the archive holds is the manifest's closure plus whatever is left.
The closure is the manifest, every `lib` and `bin` file it names, every
module its lib SQL declares, and `README.md`; those ship whatever the
ignore rules say, which is what lets a build directory be excluded
while the built wasm inside it travels. Everything else ships unless it
is excluded: entries whose name starts with `.` never ship, and
`.ffrwdignore` and `.gitignore` at the package root add to that. Both
are read when both are there, and their patterns union.

The patterns are a gitignore subset - blank lines, `#` comments, a bare
name matching at any depth, `dir/` for a directory's whole subtree, a
leading `/` anchoring to the package root, `*` within a path segment
and `**` across them. Matching is case-sensitive. There is no
negation: the closure is what pulls a file back, so nothing needs one.
A `!` line in `.ffrwdignore` is refused, naming it; a `!` line - or any
other line outside the subset - in a borrowed `.gitignore` is skipped
with a warning instead, since that file was written for another tool.

`README.md` is rendered from CommonMark to HTML at publish and stored
on the version as `readme_html`, which is what the site shows. Raw HTML
in it passes through; the site sanitizes before it inserts anything.
Publishing without one warns rather than refuses.

What the registry checks is only what it alone can: the token and its
scope, that the namespace is yours or unclaimed, that a version already
published is not being changed under the same name, and that the bytes
hash to the claimed digest. Its refusals are printed as their own
message and hint. Republishing a version with the same bytes succeeds
and changes nothing; republishing it with different bytes is refused -
a published version may not change, so bump the version instead.

The version's visibility is the manifest's `private` key (see
[Projects and packages](#projects-and-packages)): true publishes it
private, absent or false public. There is no flag; the manifest is the
one place visibility is written.

## FROM items

Every FROM item is a compile-time table; the column model per shape is
[rows.md](rows.md), the type vocabulary [types.md](types.md).

| form | rows | notes |
| --- | --- | --- |
| `input('path', name => value, ...) alias` | 1, or one per rendition of an HLS/DASH manifest | alias mandatory; path is a literal, never computed; trailing named options are ffmpeg's per-input flags; a manifest is a row table ([rows.md](rows.md#rendition-rows---inputladderm3u8-r)) |
| `ffmpeg.<source>(name => value, ...) alias` | 1 | generated stream (testsrc2, sine, color, anullsrc, ...), no `-i`; options named-only |
| `<pkg>.<source>(<values>) alias` | one per rendition of its catalog | a `RETURNS source` wasm function, probed at compile time; arguments are values only; reads like a manifest input, and a source reporting itself unbounded is live. Over a values-world export it is invoked at compile time instead: each row it answers names a `url` ffmpeg opens with its own `-i`, and the alias still reads as rendition rows |
| `unnest(alias.<array>) alias` | one per element | the four stream arrays, or `chapters` / `cues` / `embeddings` / `attachments`, of an input declared earlier in the same FROM; `cues['title']` and `embeddings['title']` name one track by its title |
| `unnest(ARRAY[STRUCT(v AS c, ...), ...]) alias` | one per array element | a written row table; columns are the STRUCT field names, every element declaring the same set |
| `generate_series(start, stop[, step]) alias` | `stop - start` over `step`, inclusive | alias mandatory, names both the row table and its one column (`i.i`); bounds and step are integer literals after substitution |
| `cte_or_view_name [alias]` | its body's rows | a multi-row body is a multi-row source |
| `function_name(args) alias` | its body's rows | a table-returning function, expanded at compile time |

Comma between items is a cross join with real multiplicity.
`JOIN ... ON` exists between two row tables - `unnest` tables (chapter
rows included), CTEs and views, struct row tables, `generate_series` -
and nowhere else: `INNER`, `LEFT [OUTER]`, `FULL [OUTER]`, each with
its own `ON`. An outer join's gap side has NULL streams; fill with
`COALESCE` and a generated source ([rows.md](rows.md#joins)), or
select the gaps at a manifest destination, where they mean absence
(see Destinations and options).

### input() options

`input('path', name => value, ...)` takes trailing named options after the
path. They are ffmpeg's own per-input flags, rendered immediately before that
input's `-i`, in the order written.

- The path is the only positional argument; anything else positional is
  rejected, with the named spelling in the hint.
- Values are compile-time literals, checked against the option's declared
  type. A NULL means the option is not written.
- The options that shape what the demuxer reads (`format`, `framerate`,
  `video_size`, `pixel_format`, ...) reach the **probe** too, in the same
  order the decode gets them, so what a query reads back about the input
  (`*`, stream counts, durations) is what ffmpeg will see. Options that only
  shape decode or playback pacing (`realtime`, `stream_loop`, `hwaccel`,
  `itsoffset`, `seek_end`) do not — ffprobe has no use for them, and some
  reject them outright.
- The probe is cached per `(path, options)`. One path read two ways is two
  `-i` entries and two probes; two aliases spelling the same path *and* the
  same options share one `-i` and one probe.

`format` is the option that changes what the path means: with a demuxer
named, the path is that demuxer's to interpret and need not exist on disk.
That is what makes capture devices and synthetic sources reachable.

```sql
COPY (SELECT a.video[1] FROM input('video=Logitech BRIO', format => 'dshow',
  framerate => 30, video_size => '960x540') a) TO 'cam.mkv'
```

Protocol options (`rtsp_transport`, `user_agent`) are the same mechanism on a
URL. A live source has no duration for the probe to report and behaves like
any other input whose duration is unknown.

## The SELECT list

Each column is one of:

- **A stream**: `f.video[1]`, a bare array splat (`f.audio` = every
  track), a bare track-row alias (`a`, the row IS the stream), `*`, or
  a filter call over any of these. In a media COPY, column order is
  `-map` order.
- **A filter call**: any filter of the installed ffmpeg, bare or
  `ffmpeg.<name>`, plus the `ffrwd.<name>` macros - streams first,
  then options positionally in the filter's own order, then
  `name => value` ([filters.md](filters.md)). Bare arrays broadcast;
  two arrays in one call zip elementwise. `VARIADIC <array>` is a third
  reading: a trailing, at-most-one argument that spreads the array as
  the pad list instead, for a filter whose pad count follows its
  argument count (the N-input set, `concat`) - `amix(VARIADIC f.audio)`
  or `concat(intro, VARIADIC array_agg(v))`. An explicit count option
  that disagrees, or an empty array, is a rejection; a fixed-arity
  filter does not take `VARIADIC` at all.
- **A `tags` column**: a column named `tags` holding a map. Over track
  rows its keys land on the row's streams; over input rows only, on the
  container; a `NULL` field clears its key ([rows.md](rows.md#tags)).
  It is the ONLY column that writes metadata.
- **A `disposition` column**: an aliased value that sets the row's
  flags rather than a tag.
- **A value column** (CTE bodies): any other aliased compile-time
  value becomes a column of the body's rows, readable downstream. At a
  media sink such a column is a rejection - a SELECT column there is an
  output stream.
- **`array_agg(<per-row stream expression>)`**: gathers rows in row
  order; must be a whole column, or the sole argument of `VARIADIC`
  ([rows.md](rows.md#combining-rows)).
- **`ARRAY(<select>)`**: gathers a single-column, countable subquery's
  rows into an array, in expression position - the converse of
  `unnest`, and everywhere an `array_agg` result already stands
  (a whole column, or `VARIADIC`'s argument). `SELECT AS STRUCT
  <cols>` gathers a multi-column subquery into an array of structs
  instead, feeding a `chapters` / `attachments` column or a cue array
  the way `array_agg(STRUCT(...)::<record>)` does by hand. The
  subquery is self-contained (its own FROM, no reference to a row
  source outside it) and this branch may have no row source of its
  own already.
- **A metadata column** (table queries): any row column prints as
  data.
- **`*` / `<alias>.*`**: over an input, its array columns - the four
  stream arrays in `video`, `audio`, `subtitle`, `data` order in a
  media query, every array column including `chapters` in a table
  one. Over rows, the record's scalar fields (`tags` and `disposition`
  excluded, read them by name), which a table query prints and a media
  query rejects. Over a CTE, the stream columns its body named.

  `* EXCEPT(name, ...)` drops the named columns from the expansion;
  `* REPLACE(expr AS name, ...)` keeps the expansion's order but
  produces `name`'s slot from `expr` instead - a media query only. A
  name is a kind (`video`/`audio`/`subtitle`/`data`) over an input or a
  generated source, or the column name a CTE gave it. A name may
  appear in EXCEPT or REPLACE at most once, and a name absent from
  this file (a kind with no streams) is a no-op, exactly like a bare
  `*` skipping it. `REPLACE`'s `expr` does not have to keep the slot's
  original kind - the same freedom an ordinary aliased SELECT column
  already has.

Subscripts are positive integer literals, 1-based.
`(f.audio[1]).codec`-style accessors reach row columns without
unnest; in WHERE they are assertions. A tag is read by path,
`f.tags.title` / `t.tags.language`, one key at a time, and so is a
disposition flag, `t.disposition.forced`, over a closed key set. A bare
`f.tags` is the whole map: no value on its own, but an operand of `||`
in a `tags` column.

## Values and predicates

The borrowings below are instances of the split named at the top of
this page: value-model spellings come from BigQuery, and each is
recorded here at its point of use.

`STRUCT(value AS name, ...)` is a **deviation**: Postgres has no such
literal, and the spelling is borrowed (BigQuery's). It is the dialect's
one way to write a map or a record by field name — the `tags` column
takes a map, and a `::chapter` / `::cue` / `::attachment` cast turns one
into that record.

`* EXCEPT(...)` / `* REPLACE(...)` are borrowed the same way (BigQuery's
splat modifiers); `EXCEPT` is otherwise a Postgres set operator, but the
parenthesized form only ever appears after a bare `*`, where set
subtraction has no meaning.

`SELECT AS STRUCT <cols>` inside `ARRAY(...)` is borrowed too (BigQuery's
struct-valued SELECT); it names the array-of-structs form of a gathered
subquery, since a plain multi-column SELECT there is a typed rejection.
`ARRAY(<select>)` itself is not a borrowing - Postgres has it natively,
and this dialect's `unnest(ARRAY[STRUCT(...), ...])` row table is its own
addition, not a borrowed spelling.


One compile-time value grammar serves predicates, `tags` fields, value
columns, trim bounds, computed filter arguments, and fan-out
destinations:

```
value := literal | NULL | row-column | input-scalar
       | value || value            -- text only
       | value (+|-|*|/) value     -- Postgres typing; int/int truncates
       | value ::text
       | CASE WHEN pred THEN value [ELSE value] END
       | COALESCE(value, ...)      -- first argument a value, never a stream;
                                   -- arguments agree on one type
       | function(value, ...)      -- upper, lower, length, round, replace,
                                   -- substring, or a value wasm function;
                                   -- over a row column, once per row

       | :'var' | :"var" | :var    -- CLI -v substitution, psql's forms
       | :var[k] | :'var'[k]        -- one element of a comma-split -v list
       | ARRAY[literal, ...][k]     -- an array element; the subscript is a
                                    -- positive integer literal or a number
                                    -- row column, picked per row
       | STRUCT(value AS name, ...)          -- a map, or a record with a cast
       | map || map                          -- merge, right side wins
       | ARRAY[STRUCT(...)::chapter, ...]    -- record arrays: chapter,
       | ARRAY[STRUCT(...)::cue, ...]        -- cue, attachment
```

`::text` is the spelling; `CAST(value AS text)` compiles too, but only
because sqlglot 30.17 parses it to the identical node with no marker
telling the two apart, so it is an undocumented synonym rather than a
second supported spelling.

Predicates: `= != < <= > >= BETWEEN IS [NOT] NULL [NOT] IN (literals)`,
combined with `AND OR NOT`. A boolean value is a predicate on its own
(`WHERE t.disposition.default`). All decided at compile time against probed
metadata - never a runtime ffmpeg predicate. NULL follows SQL:
`=`/`!=` both fail against it.

`WHERE alias.t BETWEEN a AND b` (either bound alone also works) is the
trim window - it compiles to seeks, not filters
([trimming.md](trimming.md)). Bounds take the value grammar, including
`f.duration` and any row column. A bound reading a row column is one
window per row, and the query says where those rows go: a fan-out
`TO (expression)` gives each a file, an aggregate gathers them into one.

## Grouping and combining

A single destination takes exactly ONE row; a multi-row relation
combines only when written (`array_agg` + `GROUP BY`) or fans out
(`TO (expression)`, one file per row/group). The four rules, the
resolved-row-count principle, and grouped fan-out are in
[rows.md](rows.md#combining-rows). GROUP BY, ORDER BY, LIMIT and
OFFSET are legal only over row-table queries; Postgres's grouping rule
is enforced. LIMIT and OFFSET narrow the resolved row set after WHERE
and ORDER BY and before the one-row rule, so `ORDER BY t.width DESC
LIMIT 1` is the top row with no aggregate. Their counts are integer
literals after `-v` substitution (the generate_series rule); `LIMIT 0`
is rejected - a query that selects nothing is a mistake worth naming -
and so is an OFFSET that skips every row.

## Destinations and options

`TO 'path'` writes one file; `TO STDOUT WITH (format 'csv')` prints;
`TO (value-expression over row columns)` writes one file per row or
group; `TO <sink>(<values>)` hands the relation's ROWS to a
`RETURNS sink` wasm function, which writes no file at all: one row is
a file-shaped sink, N rows a ladder, a NULL cell that kind absent
from the rendition - the manifest destination's reading, so a ladder
read from one manifest republishes through a sink with no
aggregation.

`WITH (format 'hls')` and `format 'dash'` make the destination a
**manifest** - the third answer to a multi-row relation, beside the
one-row path and the fan-out. The written name is the master playlist
(`.m3u8`) or the `.mpd`; the outputs it binds are variant playlists
and segments, so the relation stays rows, each row one entry of the
variant map, in row order. A NULL stream cell (an outer join's gap)
means that kind is absent from the variant: a video-only row is a
variant drawing from the audio group, an audio-only row a rendition in
it, a both-cells row a muxed variant. Per-row `WITH` options bind per
row as they do under a fan-out, a NULL read meaning the encoder's own
default. The same rows into `TO (expression)` are N files; into a
manifest they are one ladder ([recipe
104](examples.md#104-publish-the-ladder-as-hls)).

Under a manifest format the compiler owns what the format needs:

- **Alignment.** The keyframe interval is derived from the segment
  length (`hls_time` / `seg_duration`, the muxer's default when unset)
  and the frame rate (`fps()` when the query writes one, probed
  otherwise): gop, `keyint_min` pinned to it, scene cuts disabled in
  the encoder's own spelling. An explicit `gop` that does not divide
  the segment is refused, naming the nearest ones that would.
- **Layout** (hls). The destination names the master; variant
  playlists and segments are laid out under `%v` directories beside
  it, the init segment named - every path consistent, each overridable
  through `master_pl_name` / `hls_segment_filename` /
  `hls_fmp4_init_filename`, none required. dash's muxer already writes
  everything beside the `.mpd`.
- **The variant map.** `var_stream_map` / `adaptation_sets` is a
  transcription of the rows - `v:N`/`a:N` in output order, an `agroup`
  binding the demuxed shape, names from the streams (height for video,
  language tag for audio, positional `a0` when `und` or colliding),
  `default:yes` on the first rendition unless a probed default
  disposition says otherwise. A hand-written map is refused, naming
  what the compiler would write.

The format-specific options (`hls_time`, `hls_playlist_type`,
`hls_flags`, `hls_segment_type`, `hls_segment_filename`,
`hls_fmp4_init_filename`, `master_pl_name`; `seg_duration`,
`use_template`, `use_timeline`, `init_seg_name`, `media_seg_name`,
`single_file`) are refused outside their format. Sink options (`WITH
(...)`) cover codecs, quality, bitrate control, metadata copying,
two-pass - the full table is generated into the prompt (`ffrwd
prompt`) and validated per option with typed errors; a sink function
destination takes none.

An option value is a literal or `ARRAY[literal, ...][k]` - what
`:'var'[k]` substitutes to - and nothing else: it is settled before
ffmpeg runs, so a column off the media is a rejection naming the
option. A subscript reading a row column picks per file, so a fan-out
`TO` may vary its encode per rung ([recipe
103](examples.md#103-give-each-rung-of-the-ladder-its-own-encode));
under a quoted `TO` there is no row to read, and that is a rejection
too.

## Variables

`:'name'` (string literal), `:"name"` (identifier) and bare `:name`
(raw text) are psql's reference forms, filled by `-v name=value`. An
UNSET reference substitutes to the bare keyword `NULL` — never `''`,
never the literal text, never an error. That is a deviation from psql,
which leaves the text alone.

**NULL is absence.** A NULL — an unset variable's, or written literally
— in an option position means the option is not written and the thing
being configured supplies its own default. This holds at every binding
site: filter options (positional or named, `enable` included), source
filter options, `input()` options, and `COPY ... WITH (...)` options. A
dropped positional still occupies its slot: `scale(v, :w, :h)` with
`:w` unset writes only `height`, nothing shifts, and a named repeat of
the dropped option still collides.

What is required is derived from use, and a NULL there is a
compile-time rejection naming the variable when the NULL came from one
(`':source' was not set`):

- `input(NULL)` — a path is required.
- `COPY ... TO NULL`, and a `TO (expression)` that evaluates to NULL
  for a row — a destination is required.
- a NULL in a stream position of any call.
- a NULL for an option on the required list below — where omitting the
  option entirely is the same rejection, since ffmpeg's own init()
  would refuse it at run time.

The required list is hand-kept (ffmpeg has no required-option
metadata):

| filter | required |
| --- | --- |
| `subtitles` | `filename` |
| `lut3d` | `file` |
| `frei0r` | `filter_name` |
| `ladspa` | `file`, `plugin` |
| `movie`, `amovie` | `filename` |
| `drawtext` | `text` or `textfile` (either satisfies) |
| `xfade` | `expr`, only when `transition` is `custom` |

Everything else falls through to ffmpeg's own error at run time.

**A function parameter's `DEFAULT` reads NULL as absence too — a
deviation from Postgres.** In Postgres, NULL is a value and only an
omitted argument triggers a `DEFAULT`; here, an explicit NULL argument to
a defaulted parameter takes the `DEFAULT` the same way omitting it does,
because NULL is absence everywhere else in the dialect and calls are
otherwise positional, so a caller who always writes the argument (an
unset variable, say) has no other way to omit it. A parameter with no
`DEFAULT` still takes a written NULL unchanged — that stays the way to
mean NULL itself.

One place absence is not "leave alone": a NULL field of a `tags` column
clears that key, so `STRUCT(:'title' AS title) AS tags` unset clears the
title. A program that means "keep unless told otherwise" writes
`STRUCT(COALESCE(:'title', f.tags.title) AS title) AS tags` — ordinary
SQL.

**Lists.** Every reference form takes an optional subscript,
`:name[k]`: the value splits on commas and the reference substitutes to
ONE element, 1-based, quoted the way the form asks (`:'name'[k]` a
string literal, `:name[k]` raw text, `:"name"[k]` an identifier). A
literal subscript resolves at substitution time; a subscript past the
end is a rejection naming the list's length. The subscript may also be
a row column (`:widths[i.i]`) - then the reference substitutes to an
`ARRAY[...]` element access (raw elements for `:name`, string literals
for `:'name'`) and the element is read per row during lowering, under
the same static rule subscripts have everywhere. An identifier is a
compile-time name, so `:"name"[...]` takes a literal subscript only.
Unsubscripted, a comma-carrying value stays the one raw text it always
was - splitting happens only where a subscript asks for it. The whole
variable unset stays NULL-is-absence, subscripted or not.

The check on `-v` points the other way: since an unset reference is
legal, `-v name=value` for a name the text never references is the
usage error (exit 2), naming the names the text does reference. The
`-- variables:` header remains documentation, not a declaration.

## Not in the dialect

Every one of these is a typed rejection, never a silent reinterpretation:

- **Statements**: anything but SELECT / COPY / CREATE VIEW; more than
  one bare statement; INSERT/UPDATE/DELETE/DDL.
- **Subqueries** anywhere except CTE and view bodies - `IN (SELECT
  ...)`, `EXISTS`, derived tables in FROM.
- **Joins**: `RIGHT [OUTER] JOIN`, `CROSS JOIN` (spell it with a
  comma), `NATURAL JOIN`, `USING`, and any `JOIN ... ON` not between
  two row tables.
- **No streaming equivalent**: `HAVING`, `DISTINCT`, `UNION` without
  `ALL`, window functions, `QUALIFY`, aggregates other than `array_agg`
  (`count`, `sum`, ...), `ORDER BY` inside `array_agg`; `LIMIT` and
  `OFFSET` outside a row-table query (see Grouping and combining).
- **Aggregation context**: GROUP BY / array_agg inside a CTE body, a
  view body, or a UNION ALL branch; a per-stream `tags` column in a
  grouped query (tag inside a CTE, aggregate outside).
- **Values**: casts other than to text; computed input paths;
  computed subscripts; `0` or negative subscripts; `||` over numbers
  without `::text`; division by a known zero; a vector compared,
  concatenated, cast to text, or written as a tag; `cos_similarity`/
  `vector_length` called with the wrong argument count, or over
  anything but a vector - a length mismatch between the two
  `cos_similarity` arguments is refused too, but only once the vectors
  themselves are known, naming both lengths.
- **`generate_series`**: a bound or step that is not an integer literal
  after substitution (a column reference included); a `0` step; a
  descending or empty range; an unaliased call.
- **Multi-row into one path** (`ROW_COUNT_MISMATCH`): gather or fan
  out, explicitly.
- **Filters**: variable-OUTPUT-pad (`split` - what the compiler's own
  split pass is for; UNION ALL is `concat` without ever naming it),
  multi-output (`scale2ref`, `feedback`), sinks, multi-output sources
  (`movie`, `avsynctest`); options typed `binary` or `dictionary`;
  runtime filter commands (`sendcmd`, `zmq`). A variable-INPUT-pad
  filter (`amix`, `hstack`, `xstack`, and every other filter your
  ffmpeg reports that way) is an ordinary callable filter, taking any
  stream count positionally or under `VARIADIC`. `ffmpeg.concat`/
  `concat` takes any stream count too, but ONLY under `VARIADIC` -
  called without it, `concat` is still `UNKNOWN_FUNCTION` (its own pad
  count is variable on the OUTPUT side too).
- **Functions**: `OR REPLACE`, `IF NOT EXISTS`, a schema-qualified
  name, any property but `RETURNS`/`LANGUAGE`, a language other than
  `sql` or `wasm`, `OUT`/`INOUT`/`VARIADIC`/`COLLATE` on a parameter,
  a parameter without
  a `DEFAULT` written after one that has one, overloading, recursion,
  a body with its own `WITH` or `GROUP BY`/`ORDER BY`/`LIMIT`, a body
  referencing anything but its parameters and its own `FROM` aliases, a
  definition in the query's own text that nothing calls, and a
  `TABLE`-returning call in the `SELECT` list.
- **`LANGUAGE wasm`**: a `TABLE` return; a stream signature that is not
  one or more streams in and one of the same kind out - no stream
  parameter, a stream after a value parameter, stream parameters of
  different kinds, or a `video_stream` in and an `audio_stream` out; a
  stream parameter count the module's own does not match, or a module
  reading several streams that declares the per-frame interface or a
  window or stride other than 1; two streams into one module that do
  not run in lockstep; a
  declared kind the module does not filter; a value signature with a
  parameter or a `RETURNS` outside `vtype`, or a `DEFAULT` on one of its
  parameters; an
  export name the module does not have, or - for a value function - not
  in the module's own function list; a wit world this ffrwd does not
  host; a module declaring both pixel formats and sample formats, or
  one whose formats and the wire's do not overlap; a
  parameter the module's schema does not declare, or a value of the
  wrong type for one, on either side of the call; a value function's
  `RETURNS` that does not match the module's result type, or a module
  answering with the wrong JSON type; a named argument; a call in
  `FROM` unless it returns source, a `RETURNS source` call anywhere
  but `FROM` or one handed a stream; a `RETURNS sink` call anywhere
  but the `TO` position, a non-sink function written there, `WITH`
  options or a star SELECT list on a sink destination; a rows function
  with more than one parameter, a value parameter, or a `DEFAULT`, one
  over a module that reads no rows, one handed a stream or a
  compile-time row array, or a stream function handed rows; a URL
  source (a `RETURNS source` over a values-world export) answering no
  rows, a row naming no `url`, a row naming `width` or `height`, or
  rows that disagree on their own columns; and a module path in a
  package's lib file that leaves the package directory.
- **Annotations**: a `RETURNS STRUCT` that is not one stream field
  followed by one annotation array; an annotation field typed anything
  but `vtype`; a record the producing module's
  row schema does not match; an annotation return over a module that
  emits no rows; an annotation column anywhere but right after the
  stream, beside several streams, given a `DEFAULT` other than `NULL`,
  or defaulted on a per-frame consumer; a call returning annotations that
  nothing reads; a call taking them written over an argument that
  produces none, unless the column defaults; and two annotation
  records that disagree.
- **Projections**: a field read off a wasm call that returns no struct;
  the stream half of one read back anywhere but beside the same call's
  rows; a field the return does not declare; a `.ndjson` destination
  whose query selects anything but the one annotation column; and a
  language code at the call that no container tag stands for.
- **Runtime gathers**: a gather over a module's rows that selects
  anything but the whole row, carries a clause other than `FROM` and
  `WHERE`, or unnests without an alias; a predicate naming a field the
  record does not declare, qualifying one with anything but the
  gather's own alias, comparing a field against a value of another
  type, or holding anything outside `=`, `<>`, `<`, `<=`, `>`, `>=`,
  `AND`, `OR`, `NOT` and parentheses; and a written annotation column
  whose producing call is not the one the stream argument names.
- **Packages**: a namespace no package claims; a namespace with no
  package by that name; a two-segment call on a package whose `lib` is
  a map, naming its exports instead; a two-segment call naming an
  installed `ffrwd/<package>`, naming the three-segment form instead; a
  member the package does not export; a manifest that is not one JSON
  object with `name` and `version`; a name that is not
  `<namespace>/<package>` in plain identifiers, whose namespace is
  reserved, or that is `ffrwd/<macro name>`; a `bins` or `libs` key
  (`lib`/`bin` replaced them, its hint naming the singular); a `lib` or
  `bin` value naming a pattern, a missing file, or a path outside the
  project; an exported name not defined in the file named for it; one name defined twice
  across a package's lib files; a lib file holding anything but
  `CREATE FUNCTION`; a recipe name that is not a command word or is
  declared twice; a dependency key that is not a package name or names
  one ffrwd keeps; a dependency value that is not a non-empty
  string; a dependency cycle, naming the loop.
- **Lockfiles**: a lockfile that is not one JSON object with its three
  required keys, or is written in another format version; an entry of
  no known kind, missing a key, or holding an unknown one; two entries
  pinning one package at one version; a lockfile claiming to be
  reproducible while linking a directory; a links file that is not one
  JSON object of its known keys, or a link record naming neither a
  package nor a directory; a linked name nothing on this machine links;
  a linked directory with no manifest; a linked package whose own
  lockfile does not cover its manifest's dependencies (the hint names
  `ffrwd link` run there again); stored content that is missing or was written by another
  store layout; a downloaded archive that does not hash to what the
  entry pins, or that holds a member outside the package root, a link,
  a device, or more members or bytes than the caps allow; an entry the
  package it points at disagrees with; a `dependencies` map -
  the lockfile's own, or one carried on a registry entry - that is not
  an object of name to version.
- **The registry**: a registry that cannot be read; a detail document
  that is not JSON, is not an object, is written in another format
  version, describes another package, or holds a malformed field or a
  name that is not `<namespace>/<package>`; a package or a version the
  registry does not publish; an archive a token does not authorize
  downloading, or that this machine has no token for; an archive that
  is not the size or the digest its detail document records; an archive
  whose package says it is a different name or version than what was
  published; a pinned model that names no export of the package, will
  not download, or does not hash to its pin.
- **Publishing**: a credentials file that is not this format or holds
  no token; publishing with no token at all; a package whose exports,
  modules, models or recipes do not pass the local preflight; a
  dependency the registry cannot resolve; an archive over the size the
  registry takes; and the registry's own refusals, printed with the
  message and hint it sent.
- **Identifiers**: double-quoted identifiers (except tag-key aliases);
  the reserved names `ffmpeg` and `ffrwd` as aliases.
- **Written records**: a chapter whose span ends at or before it starts,
  or whose chapters overlap or run out of order (cues may overlap, but
  must still be ascending); an attachment with no `path`; reading
  `a.path` back.
- **Timeline**: `WHERE t` on generated sources (give the source its
  own `duration`); selecting chapter rows as streams; a bare
  `f.chapters` in a media query, or subscripting it (`unnest` it); a data/subtitle
  track through any filter (passthrough only).
- **Fields**: reading one off a filter output (`scale(v, 640, -2).width`,
  `volume(a, 0.2).tags.language`), since nothing probed it; setting a
  read-only one (`'h264' AS codec`, `3 AS index`, `12 AS duration`),
  since it is a probed fact and not an assertion; `SELECT *` over rows
  in a media query, since a star expands fields and a SELECT column is
  a stream.
- **Written chapters**: a chapter that ends at or before it starts, rows
  out of ascending order, or two chapters covering the same second.

What a specific rejection looks like, with captured JSON for every
error code: [errors.md](errors.md).
