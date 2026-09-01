"""PTS-reset pass for ffrwd IR graphs.

``trim``/``atrim`` preserve the source's own timestamps: a stream trimmed
to start at 90s still carries frames whose PTS starts at 90s. Left alone,
that is a silent correctness bug -- two trims concatenated leave a gap the
length of the first one's offset, and one trim written to a file alone
carries a lead-in of however far into the source it began. Neither errors;
both play wrong.

This pass splices a ``setpts``/``asetpts`` node (``PTS-STARTPTS``, matching
the type of stream the trim carries) immediately in front of every consumer
of a ``trim``/``atrim`` node's output, UNLESS that consumer already IS a
``setpts``/``asetpts`` -- an author, or a macro expanding to one
(``ffrwd.speed``, see ``ffrwd/macros.py``), has already taken control of
that stream's timing, and a second reset would be redundant. The check is
per CONSUMER: one trim feeding both a bare output and an author's own
``setpts`` gets a reset spliced in front of the first and nothing in front
of the second.

This is exactly the shape ``lower.py``'s own WHERE-clause CTE trim already
uses (a ``trim``/``atrim`` node immediately followed by its own
``setpts``/``asetpts``, see ``_trim``) -- that path is already protected and
this pass is a no-op on it. It exists for the OTHER way a trim reaches the
graph: an explicit ``ffmpeg.trim(...)``/``ffmpeg.atrim(...)`` call, which
lowers like any other filter and gets no such treatment on its own.

Modeled on ``split.py``'s pass exactly: one memoized reset node per
trim/atrim ref, inserted immediately before its first unprotected consumer,
which keeps `nodes` topologically ordered the same way a split does. Runs
BEFORE `insert_splits` (`compiler.py`) so the split pass sees the final
topology -- a reset node with more than one consumer is fanned out like any
other.

Pure: returns a new Graph, never mutates `g`. A graph with no
``trim``/``atrim`` node is returned byte-identical (structurally: same
nodes, same refs, nothing added).
"""

from __future__ import annotations

from .ir import FrameRef, Graph, Node, Output, SinkUnit, StreamType

# The filter names this pass reacts to, and the reset it inserts for each.
_TRIM_FILTERS = frozenset({"trim", "atrim"})
_RESET_FILTER: dict[StreamType, str] = {"video": "setpts", "audio": "asetpts"}
_RESET_EXPR = "PTS-STARTPTS"


def _trim_node(g: Graph, ref: FrameRef) -> Node | None:
    """The trim/atrim node `ref` names, else None.

    `ref` is never pad-qualified for one: `trim`/`atrim` always mint exactly
    one output pad, so a source ref or another node's ref both correctly read
    as "not a trim" here without any extra parsing.
    """
    node = g.nodes.get(ref)
    return node if node is not None and node.filter in _TRIM_FILTERS else None


def insert_pts_resets(g: Graph) -> Graph:
    """Splice a PTS reset in front of every unprotected trim/atrim consumer.

    Pure: returns a new Graph; `g` is left unmodified.
    """
    new_nodes: dict[str, Node] = {}
    reset_ids: dict[FrameRef, str] = {}

    def rewire(ref: FrameRef, *, protected: bool) -> FrameRef:
        trim = _trim_node(g, ref)
        if trim is None or protected:
            return ref
        reset_id = reset_ids.get(ref)
        if reset_id is None:
            stream_type = trim.outputs[0]
            reset_id = f"{ref}_pts"
            reset_ids[ref] = reset_id
            new_nodes[reset_id] = Node(
                id=reset_id,
                filter=_RESET_FILTER[stream_type],
                args={"expr": _RESET_EXPR},
                inputs=[ref],
                outputs=[stream_type],
            )
        return reset_id

    for node in g.nodes.values():
        # A setpts/asetpts consumer has already taken control of timing on
        # every edge feeding it -- including one from a trim/atrim node.
        protected = node.filter in _RESET_FILTER.values()
        new_inputs = [rewire(ref, protected=protected) for ref in node.inputs]
        new_nodes[node.id] = Node(
            id=node.id,
            filter=node.filter,
            args=dict(node.args),
            inputs=new_inputs,
            outputs=list(node.outputs),
            reads_annotations=node.reads_annotations,
        )

    # A trim/atrim mapped straight to an output file, with no filter in
    # between, is never protected -- there is no node there to have written
    # the reset.
    new_sinks = [
        SinkUnit(
            outputs=[
                Output(
                    ref=rewire(output.ref, protected=False),
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
        # Not filtergraph shape -- properties of the output files, the `-i`
        # entries and the module nodes, which keep their ids, and the seek
        # path this pass never touches -- so they pass through untouched,
        # already validated.
        input_trims=dict(g.input_trims),
        input_options={alias: dict(options) for alias, options in g.input_options.items()},
        rows_sinks=dict(g.rows_sinks),
        module_sinks=list(g.module_sinks),
        packet_sinks={
            name: [dict(pad) for pad in pads]
            for name, pads in g.packet_sinks.items()
        },
    )
