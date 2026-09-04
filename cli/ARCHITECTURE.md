# Architecture

What the compiler's stages are, what each may know and decide, the
invariants every change preserves, how to compose a change so it lands
in one place, and the failures that produced each rule. Read it before
changing the compiler; the rules are short because the failures behind
them were not.

## The stages

A query moves through these in order. Each stage knows more than the
one before and may decide only what its knowledge supports.

| stage | file | knows | may decide | must not |
|---|---|---|---|---|
| parse | `parser.py` `parse` | the text | that it parses | anything else |
| rewrite | `vars.py`, `functions.py` | variables, declarations, views | the text after substitution and inlining | admissibility |
| resolve | `parser.py` `resolve` | names, types, packages | that every name binds and every type agrees | anything that needs a probe |
| probe | `probe.py` | files, manifests, sources | nothing; it reports | shape |
| plan | `lower.py` | everything above | the graph: it runs the relational half and emits streams | how a stream is rendered |
| admit | one pass over the graph | the whole graph, probed | whether the graph is a shape this compiler supports | anything after it |
| optimize, global | `split.py`, the pts reset, pushdowns | the graph | a smaller graph that means the same | process boundaries |
| partition | `processes.py` | the graph | which process hosts what, and each process's edges | argv |
| optimize, local | `startup.py`, buffer bounds, spellability | the processes | order, depth, whether every edge is spellable | shape |
| lower | `emit.py`, `wasm.py`, `execute.py`, `pipes.py` | the processes and the platform | argv, pipes, spawn order | meaning |

Today `admit` is not one pass: 314 refusal sites live in resolve and
384 in plan, and both decide admissibility. Until it is one pass, a new
admission check goes in plan, after the probe, and never in resolve
unless it needs nothing a probe would tell it. The plan file is where
the stages collapsed as the compiler grew, and the work of extracting
them is ongoing; a new change must not add to the collapse.

## The invariants

**Everything is compile-time countable.** Every relation's size is
known before ffmpeg runs. That is what lets `WHERE` filter tracks at
compile time, what keeps the one-row rule decidable, and what makes the
graph small enough to reason about. A change that would make a count
depend on the run is a change to the language, not to the compiler,
and starts as a design conversation.

**The graph is the boundary, and the relational half has already run
by the time it exists.** Rows, `WHERE` over tracks, `GROUP BY`, the
one-row rule: all of it resolves in plan, at compile time, and only
streams cross into the graph. So the graph is not a logical plan
awaiting execution; it is the residue after the relational plan has
run. Every pass after plan is a function on that graph and may assume
it is fully known.

**One home per fact.** A fact decided in two stages will drift, because
each copy was right on the day it was written and only one of them is
revisited when the world changes. The canonical homes:

- *Admissibility*: the admit pass, on the graph, with everything known.
- *A process's outputs*: its outgoing edges, decided in partition. The
  renderer and the sidecar read that; neither counts anything else.
- *A module's effects and grants*: its own description, read once, and
  carried on the process. Compile-time and run-time ask the same
  place.
- *A track's identity*: the probe. Names, languages, renditions ride on
  the stream from the probe onward; nothing downstream re-derives them.
- *A refusal's text*: one site per code. Two sites saying the same
  thing are one bug away from saying different things.

**Passes are pure functions on the graph.** A pass takes a graph and
returns a graph, and may be tested with a synthetic one. A pass that
reaches outside its input, to the file system, the sidecar or a global,
has left the pipeline and will be hard to move.

**A refusal is typed, anchored, hinted, and honest about why.** The
code says what class of thing went wrong, the position says where, the
hint says what to do instead. It also says whether the refusal is
fundamental (ffmpeg cannot express it, or it is not compile-time
countable) or merely not built yet, so that a sweep over the query
space can separate the wall from the work without a person reading
each one.

## Composing a change

1. **Recipe first, red.** A new shape starts as a cookbook or corpus
   entry that does not compile, with the query written the way a user
   would write it. The pinned command is generated when it goes green,
   never typed.
2. **Name the owning stage before writing anything.** If the change
   needs a probe, it is not resolve's. If it changes the graph's shape,
   it goes before partition. If it needs process boundaries, it goes
   after. If it changes argv or pipes, it is lower's and touches no
   meaning. A change that seems to need two stages is usually one
   change plus a copy of a fact that already has a home; find the home.
3. **Grep for the twin before adding a check.** `_check_`, `_error(`,
   `_reject(` and the error code you are about to raise. If a check
   with the same meaning exists in another stage, the change is to
   move it, not to add a second.
4. **Read counts from the plan, never from the world.** A process
   writes its edges. A source writes the tracks the plan mapped. A
   sink reads the rows the plan gave it. A catalog, a manifest or a
   module's own list is what the probe reports, not what the run does.
5. **Test each stage where it lives.** A lowering property wants a
   synthetic probe and no ffmpeg. A rendering property wants a
   synthetic plan and no sidecar. A pass wants a graph in and a graph
   out. The exec tier and the two nets are for the whole, not for the
   part.
6. **Land against both nets.** The cookbook and corpus pin every
   command byte for byte; the shape sweep compiles every input and
   output shape against a fixture. A change that moves a pin or a
   cell is wrong until it can say, in its commit, why the old one was.
7. **Anything live is measured, not reasoned.** A live source, a
   module with a model, a pipe under load: the design says what should
   happen and the runtime does something else often enough that a
   claim about either is a measurement with a number in it, or it is a
   guess. Two runs, then a report, never a ninth attempt.
8. **CI is Linux and its ffmpeg is not yours.** A pipe, a muxer, a
   provider walk or a probe listing that behaves one way here behaves
   another there. A platform-bound test says which platform it pins.
   An ffmpeg behavior is checked under CI's own build before it is
   pushed.

## The postmortem

Each rule above was paid for. The cases, with the rule each produced.

**The star over a ladder was refused** because the check was written
for track rows, where the row is the stream and its fields are
metadata, and nobody revisited it when rendition rows arrived with
their own array columns. *A guard naming one kind of row is a guard
that will be wrong about the next kind.*

**A rendition column read through a CTE collapsed into one array**,
because the manifest path's direct read predated CTEs carrying rows,
and the CTE path re-derived the column's cardinality on its own. *A
column's cardinality follows its relation; it is decided once, where
the relation is built, and every path reads it there.* (6eea00a)

**COALESCE took only a track-row alias** because that was the only
nullable thing when it was written; a CTE's nullable stream column was
the same thing under a different syntax and was refused as a "COLUMN
expression". *A check on syntax where a check on type was meant.*

**A scalar did not broadcast over N rows** because the only multi-row
relations at the time were `unnest` and `generate_series`, whose
columns are always row sets, so the manifest destination insisted every
column be one. *Postgres's rule was the right rule all along; the
restriction recorded a date.*

**`trim`'s documented option was unreachable** because the registry
folds adjacent aliases to the longest name, which is right for
`w`/`width` and wrong for `start`/`starti`. *A heuristic over
ffmpeg's help text is checked against every filter it applies to, not
the ones it was written for.* (646af39)

**A packet source's several tracks tripped an INTERNAL** because the
partitioner exempted a source from the one-stream rule, the renderer
already wrote one output per track, and the guard between them knew
neither. *Three homes for one fact, two of them right.* (1110ceb)

**A source module got its network at compile time and not at run
time**, because the probe derived grants from the module's description
and the process plan built a source with none, while a sink's process
got them. *Compile-time and run-time ask the same place.* (71a2b54,
8e30cec)

**A source wrote every catalog track and the query mapped some.** The
sidecar insisted on one output per catalog track, the partitioner
obliged with all of them, and an unmapped pipe had no reader: it
filled, the sidecar blocked, and the stage stopped. It worked on a
four-second ladder because thirty frames fit in a pipe's buffer, and
wedged on thirty seconds. The stall detector reported it thirty
seconds later as an overflow, and that report was read as a slow model
for most of a day. *A process writes its edges. A count read from a
catalog instead of from the plan is a count that will disagree with
the plan.* Fixed in two steps: the partitioner makes a source's
outputs its edges and the sidecar writes only those (9b9fecd), then
the interface tells the source what to pull so the rest never leaves
the relay (c12937e). No compatibility path was kept: the only packet
sources that existed were rebuilt, and a rule with an exception is
two homes again.

**The slow-model diagnosis was reasoned, not measured.** The design
said a model builds its session on the first frame and that could
plausibly exceed the stall window, so pre-warming was proposed. The
model reaches its first frame in 2.6 seconds on CUDA and 6.2 on CPU.
One measurement would have saved the proposal. *Anything live is
measured.*

**An untimed run overflowed on Windows** because an infinite deadline
reached a wait that cannot take infinity there, and CI, on Linux, never
saw it. *A platform-bound path is tested on the platform it binds.*
(6ec957e)

**An audio-only HLS master probed as two rows under CI's ffprobe and
one under this machine's**, because ffprobe 8 lists a playlist shared
between a variant and its group twice and ffprobe 9 once, and the
claim logic had been checked against 9 alone. *An ffmpeg behavior is
checked under CI's own build.* (1cc0c4e)

**A refusal escaped `run` as a Python traceback** where `compile`
printed it typed, because the two entry points wrapped the same
rendering call differently. *One site per refusal, one catch per
entry point, and the entry points share it.* (64ecab0)

## Smells

Any of these in a diff is a reason to stop and find the home.

- The same predicate, or the same count, in two files.
- A guard whose comment names one kind of row, stream or process.
- A length taken from a catalog, a manifest or a module's own list
  where the plan's edges are at hand.
- A check that needs a probe, sitting before the probe.
- "as today" or "the same as X" in a comment beside a copy of X's
  logic rather than a call to it.
- A test that passes because the fixture is small.
- A claim about a live path with no number in it.
- A ninth attempt.
