"""A compiled query as a mermaid flowchart.

One pure function over what ``explain`` already computes: the IR graphs and,
for a partitioned query, the :class:`~ffrwd.processes.ProcessPlan` beside
them. Inputs and output files draw as stadium nodes carrying their path,
filters as rectangles named ``filter(key=value, ...)`` with long option lists
elided, and a split's fan-out is simply its several outgoing arrows.

A single command is one flat flowchart. A command SEQUENCE (two_pass,
loudnorm2, the copy-and-trim fan-out) draws one subgraph per command, in
order. A partitioned query draws one subgraph per process and labels each
edge between processes with what crosses it: NUT-wrapped frames, a subtitle
track, rows.

``--diagram`` renders the same text in the terminal through ``termaid``, an
optional extra imported only inside :func:`render_terminal`;
:func:`termaid_available` is what the CLI checks first, so a missing extra is
one line naming :data:`INSTALL_HINT` rather than an ImportError traceback.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence

from .ir import PIPE, FrameRef, Graph, Node, is_src, src_alias
from .processes import (
    Edge,
    ProcessPlan,
    RowsEdge,
    SidecarProcess,
    StreamEdge,
)

__all__ = ["INSTALL_HINT", "render_diagram", "render_terminal", "termaid_available"]

INSTALL_HINT = (
    "rendering a diagram needs the 'termaid' package, an optional extra: "
    'install it with `pip install "ffrwd[diagram]"`'
)

# The longest option list a filter label carries before it is elided.
_ARGS_LIMIT = 40

# What no destination-less sink can print as a path.
_NO_DESTINATION = "no destination"


def termaid_available() -> bool:
    """True if the ``termaid`` package is importable.

    Never raises: a broken install answers the same as a missing one, and
    either way the caller has a message to print rather than a traceback.
    """
    try:
        return importlib.util.find_spec("termaid") is not None
    except (ImportError, ValueError):
        return False


def render_terminal(mermaid: str) -> str:
    """`mermaid` as terminal box art. Imports ``termaid``, the optional extra."""
    import termaid

    return str(termaid.render(mermaid))


def render_diagram(graphs: Sequence[Graph], plan: ProcessPlan | None = None) -> str:
    """The flowchart for one compile: its graphs, or its process plan.

    With a `plan`, each process is a subgraph and the plan's own edges join
    them. Without one, a single graph draws flat and a sequence draws one
    subgraph per command, in command order.
    """
    lines = ["flowchart LR"]
    if plan is not None:
        lines += _plan_lines(plan)
    elif len(graphs) == 1:
        lines += [f"  {line}" for line in _graph_lines(graphs[0], "")]
    else:
        for index, graph in enumerate(graphs):
            lines.append(f'  subgraph cmd{index} ["command {index + 1}"]')
            lines += [f"    {line}" for line in _graph_lines(graph, f"cmd{index}_")]
            lines.append("  end")
    return "\n".join(lines)


def _plan_lines(plan: ProcessPlan) -> list[str]:
    """One subgraph per process, then the edges between them, labelled."""
    lines: list[str] = []
    for process in plan.processes:
        lines.append(f'  subgraph {process.id} ["{process.id}"]')
        graph = process.graph
        if graph is None:
            # A region carrying no graph of its own: one node for its module.
            # Only a sidecar may lack one; an ffmpeg process always carries a graph.
            assert isinstance(process, SidecarProcess)
            body = [f'{process.id}_m["{_escape(process.module)}"]']
        else:
            body = _graph_lines(graph, f"{process.id}_", skip_pipes=True)
        lines += [f"    {line}" for line in body]
        lines.append("  end")
        if isinstance(process, SidecarProcess):
            lines += _rows_file_lines(process)
    for edge in plan.edges:
        lines.append(f"  {edge.source} -->|{_escape(_edge_label(edge))}| {edge.target}")
    return lines


def _rows_file_lines(process: SidecarProcess) -> list[str]:
    """The rows file a sidecar writes itself, which no plan edge carries."""
    rows = process.rows
    if rows is None or not rows.path:
        return []
    rows_id = f"{process.id}_rows"
    return [
        f'  {rows_id}(["{_escape(rows.path)}"])',
        f"  {process.id} -->|{_escape(rows.container)} rows| {rows_id}",
    ]


def _edge_label(edge: Edge) -> str:
    """What crosses this edge, as the arrow's label."""
    if isinstance(edge, StreamEdge):
        return f"{edge.format.container} {edge.format.codec}"
    if isinstance(edge, RowsEdge):
        return f"{edge.container} rows"
    if edge.format.path is not None:
        return edge.format.path
    return edge.format.content


def _graph_lines(g: Graph, prefix: str, *, skip_pipes: bool = False) -> list[str]:
    """One graph's nodes and edges. `skip_pipes` drops the :data:`PIPE` ends:
    inside a process subgraph, the plan's own edges already draw them."""
    lines: list[str] = []
    taken: set[str] = set()

    input_id: dict[int, str] = {}
    for index, path in enumerate(g.input_paths):
        if skip_pipes and path == PIPE:
            continue
        input_id[index] = _claim(f"{prefix}in{index}", taken)
        lines.append(f'{input_id[index]}(["{_escape(path)}"])')

    node_id = {name: _claim(f"{prefix}{_safe(name)}", taken) for name in g.nodes}

    def source_of(ref: FrameRef) -> str | None:
        if is_src(ref):
            return input_id.get(g.sources.get(src_alias(ref), -1))
        return node_id.get(_producer(ref))

    for name, node in g.nodes.items():
        lines.append(f'{node_id[name]}["{_escape(_node_label(node))}"]')
        for ref in node.inputs:
            source = source_of(ref)
            if source is not None:
                lines.append(f"{source} --> {node_id[name]}")

    for index, unit in enumerate(g.sinks):
        if skip_pipes and unit.path == PIPE:
            continue
        sink_id = _claim(f"{prefix}out{index}", taken)
        lines.append(f'{sink_id}(["{_escape(unit.path or _NO_DESTINATION)}"])')
        for output in unit.outputs:
            source = source_of(output.ref)
            if source is not None:
                lines.append(f"{source} --> {sink_id}")

    return lines


def _node_label(node: Node) -> str:
    """``filter(key=value, ...)``, its option list elided past the cap."""
    if not node.args:
        return node.filter
    parts = [f"{key}={value}" for key, value in node.args.items()]
    return f"{node.filter}({_elide(parts)})"


def _elide(parts: list[str], limit: int = _ARGS_LIMIT) -> str:
    """`parts` joined by ``, ``, cut with ``...`` once past `limit`."""
    joined = ", ".join(parts)
    if len(joined) <= limit:
        return joined
    kept: list[str] = []
    used = 0
    for part in parts:
        cost = len(part) + (2 if kept else 0)
        if used + cost > limit:
            break
        kept.append(part)
        used += cost
    if not kept:  # the first option alone is over the cap
        return parts[0][:limit] + "..."
    return ", ".join(kept) + ", ..."


def _producer(ref: FrameRef) -> str:
    """The node id `ref` names, its ``:<pad>`` suffix stripped."""
    node_id, _, pad = ref.rpartition(":")
    return node_id if node_id and pad.isdigit() else ref


def _safe(name: str) -> str:
    """`name` as a mermaid identifier: word characters only."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


def _claim(candidate: str, taken: set[str]) -> str:
    """`candidate`, suffixed until it collides with nothing already claimed."""
    while candidate in taken:
        candidate += "_"
    taken.add(candidate)
    return candidate


def _escape(text: str) -> str:
    """`text` with mermaid's reserved characters as entities, quote-safe."""
    return (
        text.replace("&", "#amp;")
        .replace('"', "#quot;")
        .replace("<", "#lt;")
        .replace(">", "#gt;")
        .replace("|", "#124;")
    )
