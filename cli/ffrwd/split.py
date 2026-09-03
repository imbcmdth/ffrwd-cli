"""Split pass for ffrwd IR graphs.

The SQL surface is a DAG: an alias or CTE may be referenced more than once,
and each Output row counts as a consumer. FFmpeg filtergraph pads are
consume-once. This pass reconciles the two by splicing a `split`/`asplit` node
in front of every FrameRef with more than one consumer, giving each consumer
its own pad.

See ir.py for the authoritative FrameRef grammar. This pass is *a* producer of
the pad-qualified `"<node-id>:<k>"` form, not the only one: a node with more
than one entry in `outputs` (e.g. `concat v=1:a=1`) produces it upstream.

Algorithm
---------
1. Count consumers of every FrameRef: each entry in every Node.inputs, plus
   each `Output.ref` in `g.outputs` -- the UNION over sink units,
   since a pad feeding two output FILES is still consumed twice. One count
   per Output, whichever sink it belongs to.
2. For any ref consumed N > 1 times, insert a split node in front of it:
   `filter="split"` for video / `"asplit"` for audio, `args={"n": N}`,
   `inputs=[ref]`, `outputs=[type] * N`. Its id is `<sanitized-ref>_split`,
   ":" replaced by "_" ("src:a:v:0" -> "src_a_v_0_split", "n3" -> "n3_split").
3. Rewire each consumer to `"<split-id>:<k>"`, k = 0..N-1, in deterministic
   order: node insertion order, each node's inputs left to right, then
   `g.outputs` in list order (rewired last).
4. Split nodes are inserted immediately before their first consumer, keeping
   the `nodes` dict topologically ordered -- a split's own input is already
   present earlier, since the input graph is topologically ordered.

A ref's stream type is resolved from the *original* graph `g`: a source ref's
is `src_parts(ref)[1]`, a node ref's is `node.outputs[pad]` (pad 0 for the
unqualified form).

Passthrough-only exemption
--------------------------
Subtitle and data source refs are EXEMPT: they never enter the filtergraph, so
there is no pad to fan out and no filter that could (ffmpeg has no subtitle
split). A `"src:<alias>:s:<k>"`/`":d:<k>"` ref used by two Outputs passes
through untouched and emit renders the same bare `-map` twice, legal ffmpeg.
Such a ref can only appear in a sink unit's outputs (ir.py's grammar note), so
exempting it cannot leave a real pad over-consumed.

Cross-GROUP passthrough exemption
---------------------------------
The same holds for a VIDEO/AUDIO source ref that no filter node consumes and
that each sink unit maps at most once: repeating `-map 0:a:0` across two
OUTPUT FILES is as legal as repeating a subtitle map, and it keeps both files
stream-COPYING that track instead of re-encoding it through an `asplit`.

* zero node consumers AND at most one Output per sink unit -> exempt;
* anything else -- a filter node consumes it, or one unit maps it twice
  (`SELECT a.audio[1], a.audio[1]` into ONE file) -> ordinary split.

The consume-once rule for real pads (`"<node-id>[:<pad>]"`) stays strict:
two sinks reading one view's pad DO get a split.

`insert_splits` is pure (returns a new Graph, never mutates `g`) and
idempotent -- every ref in the result has exactly one consumer. Everything
outside the pad SHAPE -- each sink's `path`/`options`, `Graph.input_trims`,
`Graph.input_options` -- is copied verbatim.
"""

from __future__ import annotations

from .ir import FrameRef, Graph, Node, Output, SinkUnit, StreamType, is_src, src_parts

# Stream types a filtergraph cannot carry, and therefore cannot split.
_PASSTHROUGH_ONLY: frozenset[StreamType] = frozenset({"subtitle", "data"})


def _is_passthrough_only(ref: FrameRef) -> bool:
    """True for a subtitle/data source ref -- never a filtergraph pad."""
    return is_src(ref) and src_parts(ref)[1] in _PASSTHROUGH_ONLY


def _ref_type(g: Graph, ref: FrameRef) -> StreamType:
    """Resolve the stream type (video/audio) that `ref` carries in `g`."""
    if is_src(ref):
        return src_parts(ref)[1]
    if ":" in ref:
        node_id, pad_str = ref.rsplit(":", 1)
        pad = int(pad_str)
    else:
        node_id, pad = ref, 0
    return g.nodes[node_id].outputs[pad]


def _exempt_refs(g: Graph) -> set[FrameRef]:
    """Source refs that may stay fanned out: the bare `-map`s.

    A subtitle/data source ref always qualifies -- no filtergraph carries it.
    A video/audio source ref qualifies when NO node consumes it and no single
    sink unit maps it more than once, i.e. when every one of its consumers is
    a passthrough `-map` in a different output FILE.
    """
    exempt: set[FrameRef] = set()
    filtered: set[FrameRef] = {
        ref for node in g.nodes.values() for ref in node.inputs if is_src(ref)
    }
    for unit in g.sinks:
        seen: dict[FrameRef, int] = {}
        for output in unit.outputs:
            if not is_src(output.ref):
                continue
            seen[output.ref] = seen.get(output.ref, 0) + 1
        for ref, count in seen.items():
            if _is_passthrough_only(ref):
                exempt.add(ref)
            elif count == 1 and ref not in filtered:
                exempt.add(ref)
            else:
                exempt.discard(ref)
                filtered.add(ref)  # a within-file repeat disqualifies it here too
    return exempt


def insert_splits(g: Graph) -> Graph:
    """Splice `split`/`asplit` nodes in front of every FrameRef with fan-out > 1.

    Pure: returns a new Graph; `g` is left unmodified. Idempotent: calling
    this again on the result is a no-op.
    """
    counts: dict[FrameRef, int] = {}
    for node in g.nodes.values():
        for ref in node.inputs:
            counts[ref] = counts.get(ref, 0) + 1
    for output in g.outputs:
        counts[output.ref] = counts.get(output.ref, 0) + 1
    exempt = _exempt_refs(g)

    new_nodes: dict[str, Node] = {}
    split_ids: dict[FrameRef, str] = {}
    next_pad: dict[FrameRef, int] = {}

    def rewire(ref: FrameRef) -> FrameRef:
        if counts.get(ref, 0) <= 1:
            return ref
        if ref in exempt:
            return ref  # a repeated bare -map is legal ffmpeg (see docstring)
        split_id = split_ids.get(ref)
        if split_id is None:
            ref_type = _ref_type(g, ref)
            n = counts[ref]
            split_id = f"{ref.replace(':', '_')}_split"
            split_ids[ref] = split_id
            next_pad[ref] = 0
            new_nodes[split_id] = Node(
                id=split_id,
                filter="split" if ref_type == "video" else "asplit",
                args={"n": n},
                inputs=[ref],
                outputs=[ref_type] * n,
            )
        pad = next_pad[ref]
        next_pad[ref] = pad + 1
        return f"{split_id}:{pad}"

    for node in g.nodes.values():
        new_inputs = [rewire(ref) for ref in node.inputs]
        new_nodes[node.id] = Node(
            id=node.id,
            filter=node.filter,
            args=dict(node.args),
            inputs=new_inputs,
            outputs=list(node.outputs),
            reads_annotations=node.reads_annotations,
            # A rows edge is not a pad and never fans out: nothing to split,
            # so it rides through naming the same producer it always did.
            rows_inputs=list(node.rows_inputs),
        )

    # Units in order, each unit's outputs in list order: pad assignment is
    # deterministic across the whole graph.
    new_sinks = [
        SinkUnit(
            outputs=[
                Output(
                    ref=rewire(output.ref),
                    type=output.type,
                    name=output.name,
                    metadata=dict(output.metadata),
                    disposition=output.disposition,
                )
                for output in unit.outputs
            ],
            path=unit.path,
            options=dict(unit.options),
            tags=dict(unit.tags),
            window=unit.window,
            chapters=unit.chapters,
            metadata=unit.metadata,
            attachments=list(unit.attachments),
        )
        for unit in g.sinks
    ]

    return Graph(
        input_paths=list(g.input_paths),
        sources=dict(g.sources),
        nodes=new_nodes,
        sinks=new_sinks,
        # Not pad shape -- properties of the output files, the `-i` entries
        # and the module nodes, which keep their ids -- so they pass through
        # untouched, already validated.
        input_trims=dict(g.input_trims),
        input_options={alias: dict(options) for alias, options in g.input_options.items()},
        rows_sinks=dict(g.rows_sinks),
        module_sinks=list(g.module_sinks),
        packet_sinks={
            name: [dict(pad) for pad in pads]
            for name, pads in g.packet_sinks.items()
        },
        module_sources=dict(g.module_sources),
    )
