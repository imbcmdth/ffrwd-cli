from __future__ import annotations

import pytest

from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.ir import (
    Attachment,
    Graph,
    Node,
    Output,
    SinkUnit,
    is_src,
    src_alias,
    src_parts,
)


def _serialized_sinks(d: dict[str, object]) -> list[dict[str, object]]:
    """``Graph.to_dict()["sinks"]``, narrowed for the type checker."""
    sinks = d["sinks"]
    assert isinstance(sinks, list)
    return sinks


def _build_graph() -> Graph:
    g = Graph(input_paths=["a.mp4", "b.mp4"], sources={"a": 0, "b": 1})
    g.nodes["n0"] = Node(
        id="n0",
        filter="crop",
        args={"w": 600, "h": 200, "x": 1200, "y": 50},
        inputs=["src:b:v:0"],
        outputs=["video"],
    )
    g.nodes["n1"] = Node(
        id="n1",
        filter="scale",
        args={"w": "iw*0.5", "h": -2},
        inputs=["n0"],
        outputs=["video"],
    )
    g.nodes["n2"] = Node(
        id="n2",
        filter="overlay",
        args={"x": 20, "y": 20},
        inputs=["src:a:v:0", "n1"],
        outputs=["video"],
    )
    g.sinks = [
        SinkUnit(
            outputs=[
                Output(ref="n2", type="video", name=None, metadata={}),
                Output(
                    ref="src:a:a:0",
                    type="audio",
                    name="eng_audio",
                    metadata={"language": "eng"},
                ),
            ]
        )
    ]
    return g


def test_graph_round_trip() -> None:
    g = _build_graph()
    d1 = g.to_dict()
    g2 = Graph.from_dict(d1)
    d2 = g2.to_dict()
    assert d1 == d2


def test_graph_to_dict_shape() -> None:
    g = _build_graph()
    d = g.to_dict()
    assert d["inputs"] == ["a.mp4", "b.mp4"]
    assert d["sources"] == {"a": 0, "b": 1}

    nodes = d["nodes"]
    assert isinstance(nodes, list)
    assert [n["id"] for n in nodes] == ["n0", "n1", "n2"]
    for n in nodes:
        assert set(n.keys()) == {"id", "filter", "args", "inputs", "outputs"}
    assert nodes[0]["outputs"] == ["video"]

    sinks = d["sinks"]
    assert isinstance(sinks, list)
    assert len(sinks) == 1
    assert set(sinks[0].keys()) == {"outputs", "path", "options"}
    assert sinks[0]["path"] is None
    assert sinks[0]["options"] == {}

    outputs = sinks[0]["outputs"]
    assert isinstance(outputs, list)
    assert len(outputs) == 2
    for o in outputs:
        assert set(o.keys()) == {"ref", "type", "name", "metadata"}
    assert outputs[0] == {
        "ref": "n2",
        "type": "video",
        "name": None,
        "metadata": {},
    }
    assert outputs[1] == {
        "ref": "src:a:a:0",
        "type": "audio",
        "name": "eng_audio",
        "metadata": {"language": "eng"},
    }


def test_graph_sinks_default_empty() -> None:
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    assert g.sinks == []
    assert g.outputs == []
    assert g.to_dict()["sinks"] == []


def test_graph_outputs_is_the_union_across_sinks() -> None:
    """The derived output list is what consume-once counts over."""
    g = _build_graph()
    g.sinks.append(
        SinkUnit(
            outputs=[Output(ref="n1", type="video", name=None, metadata={})],
            path="second.mp4",
            options={"crf": 30},
        )
    )
    assert [o.ref for o in g.outputs] == ["n2", "src:a:a:0", "n1"]


def test_node_outputs_multi_pad() -> None:
    n = Node(
        id="n0",
        filter="split",
        args={"n": 2},
        inputs=["src:a:v:0"],
        outputs=["video", "video"],
    )
    d = n.to_dict()
    assert d["outputs"] == ["video", "video"]
    n2 = Node.from_dict(d)
    assert n2.outputs == ["video", "video"]


def test_is_src() -> None:
    assert is_src("src:a:v:0") is True
    assert is_src("src:pip:a:1") is True
    assert is_src("n0") is False
    assert is_src("n0:1") is False


def test_src_alias() -> None:
    assert src_alias("src:a:v:0") == "a"
    assert src_alias("src:pip:a:12") == "pip"


def test_src_parts_video() -> None:
    assert src_parts("src:a:v:0") == ("a", "video", 0)


def test_src_parts_audio() -> None:
    assert src_parts("src:a:a:2") == ("a", "audio", 2)


def test_src_parts_invalid_type_marker_raises() -> None:
    with pytest.raises(ValueError):
        src_parts("src:a:x:0")


def test_output_round_trip_metadata() -> None:
    o = Output(
        ref="src:a:a:0",
        type="audio",
        name=None,
        metadata={"language": "fra", "title": "Director commentary"},
    )
    d = o.to_dict()
    o2 = Output.from_dict(d)
    assert o2 == o


def test_ffrwd_error_to_dict_matches_spec_shape() -> None:
    err = FfrwdError(
        ErrorCode.UDF_ARG_TYPE,
        "overlay() expects (frame, frame, int, int), got (frame, varchar, int, int)",
        line=3,
        col=12,
        hint="did you mean to wrap the second argument in input()?",
    )
    assert err.to_dict() == {
        "line": 3,
        "col": 12,
        "code": "UDF_ARG_TYPE",
        "message": "overlay() expects (frame, frame, int, int), got (frame, varchar, int, int)",
        "hint": "did you mean to wrap the second argument in input()?",
    }


def test_ffrwd_error_to_dict_with_missing_fields() -> None:
    err = FfrwdError(ErrorCode.INTERNAL, "unexpected state")
    d = err.to_dict()
    assert d["line"] is None
    assert d["col"] is None
    assert d["hint"] is None
    assert d["code"] == "INTERNAL"
    assert d["message"] == "unexpected state"


def test_ffrwd_error_str_full() -> None:
    err = FfrwdError(
        ErrorCode.UDF_ARG_TYPE,
        "bad args",
        line=3,
        col=12,
        hint="try this",
    )
    assert str(err) == "line 3:12: UDF_ARG_TYPE: bad args (hint: try this)"


def test_ffrwd_error_str_no_position_no_hint() -> None:
    err = FfrwdError(ErrorCode.INTERNAL, "boom")
    assert str(err) == "INTERNAL: boom"


def test_new_v2_error_codes_exist() -> None:
    assert ErrorCode.STREAM_NOT_FOUND.value == "STREAM_NOT_FOUND"
    assert ErrorCode.INPUT_NOT_FOUND.value == "INPUT_NOT_FOUND"
    assert ErrorCode.BROADCAST_MISMATCH.value == "BROADCAST_MISMATCH"


def test_sink_error_codes_exist() -> None:
    assert ErrorCode.UNKNOWN_SINK_OPTION.value == "UNKNOWN_SINK_OPTION"
    assert ErrorCode.SINK_OPTION_TYPE.value == "SINK_OPTION_TYPE"


# ---------------------------------------------------------------------------
# SinkUnit: sink options and the multi-sink shape
# ---------------------------------------------------------------------------


def test_sink_unit_round_trip() -> None:
    unit = SinkUnit(
        outputs=[Output(ref="n2", type="video", name=None, metadata={})],
        path="out.mkv",
        options={"video_codec": "libx264", "crf": 20, "faststart": True},
    )
    d = unit.to_dict()
    assert SinkUnit.from_dict(d) == unit
    assert d == {
        "outputs": [{"ref": "n2", "type": "video", "name": None, "metadata": {}}],
        "path": "out.mkv",
        "options": {"video_codec": "libx264", "crf": 20, "faststart": True},
    }


def test_sink_unit_chapters_round_trip() -> None:
    """The chapter list stays out of the serialized shape when the query
    named none, so a unit that writes no chapters keeps the smaller shape."""
    unit = SinkUnit(
        outputs=[Output(ref="n2", type="video", name=None, metadata={})],
        path="out.mkv",
        chapters=1,
    )
    d = unit.to_dict()
    assert d["chapters"] == 1
    assert SinkUnit.from_dict(d) == unit
    assert "chapters" not in SinkUnit(outputs=[], path="out.mkv").to_dict()


def test_sink_unit_attachments_round_trip() -> None:
    """Same convention as the chapter list: a unit carrying no attached file
    keeps the smaller shape."""
    unit = SinkUnit(
        outputs=[Output(ref="n2", type="video", name=None, metadata={})],
        path="out.mkv",
        attachments=[Attachment(path="f.ttf", filename="font.ttf", mimetype="font/ttf")],
    )
    d = unit.to_dict()
    assert d["attachments"] == [
        {"path": "f.ttf", "filename": "font.ttf", "mimetype": "font/ttf"}
    ]
    assert SinkUnit.from_dict(d) == unit
    assert "attachments" not in SinkUnit(outputs=[], path="out.mkv").to_dict()


def test_an_attachment_round_trips_with_no_tags() -> None:
    unit = SinkUnit(outputs=[], path="out.mkv", attachments=[Attachment(path="f.ttf")])
    assert SinkUnit.from_dict(unit.to_dict()) == unit


def test_output_disposition_round_trips() -> None:
    """The flags are their own field, so they stay out of the metadata dict --
    and out of the serialized shape entirely when the query asserted none."""
    flagged = Output(
        ref="0:a:0", type="audio", name=None, metadata={}, disposition=("default",)
    )
    d = flagged.to_dict()
    assert d["disposition"] == ["default"]
    assert Output.from_dict(d) == flagged

    cleared = Output(ref="0:a:0", type="audio", name=None, metadata={}, disposition=())
    assert cleared.to_dict()["disposition"] == []
    assert Output.from_dict(cleared.to_dict()) == cleared

    plain = Output(ref="0:a:0", type="audio", name=None, metadata={})
    assert "disposition" not in plain.to_dict()
    assert Output.from_dict(plain.to_dict()) == plain


def test_sink_unit_container_tags_round_trip() -> None:
    unit = SinkUnit(
        outputs=[Output(ref="n2", type="video", name=None, metadata={})],
        path="out.mkv",
        tags={"title": "Cut", "artist": None},
    )
    d = unit.to_dict()
    assert d["tags"] == {"title": "Cut", "artist": None}
    assert SinkUnit.from_dict(d) == unit


def test_sink_unit_window_round_trip() -> None:
    unit = SinkUnit(
        outputs=[Output(ref="n2", type="video", name=None, metadata={})],
        path="ch1.mkv",
        window=(0.0, 1.5),
    )
    d = unit.to_dict()
    assert d["window"] == [0.0, 1.5]
    assert SinkUnit.from_dict(d) == unit


def test_sink_unit_open_window_round_trips_as_json_null() -> None:
    unit = SinkUnit(outputs=[], path="tail.mkv", window=(5.0, None))
    d = unit.to_dict()
    assert d["window"] == [5.0, None]
    assert SinkUnit.from_dict(d).window == (5.0, None)


def test_a_sink_unit_with_no_window_omits_the_window_key() -> None:
    unit = SinkUnit(outputs=[], path="out.mkv")
    assert "window" not in unit.to_dict()
    assert SinkUnit.from_dict(unit.to_dict()).window is None


def test_a_sink_unit_tagging_nothing_omits_the_tags_key() -> None:
    """Same shape the goldens pin: an empty optional field is not serialized."""
    unit = SinkUnit(outputs=[], path="out.mkv")
    assert "tags" not in unit.to_dict()
    assert SinkUnit.from_dict(unit.to_dict()).tags == {}


def test_several_sinks_with_their_paths_and_options_round_trip() -> None:
    g = _build_graph()
    g.sinks[0].path = "1080.mp4"
    g.sinks[0].options = {"video_codec": "libx264"}
    g.sinks.append(
        SinkUnit(
            outputs=[Output(ref="n1", type="video", name=None, metadata={})],
            path="720.mp4",
            options={"crf": 26},
        )
    )
    d1 = g.to_dict()
    assert [unit["path"] for unit in _serialized_sinks(d1)] == ["1080.mp4", "720.mp4"]
    assert [unit["options"] for unit in _serialized_sinks(d1)] == [
        {"video_codec": "libx264"},
        {"crf": 26},
    ]
    g2 = Graph.from_dict(d1)
    assert g2.sinks == g.sinks
    assert g2.to_dict() == d1


# ---------------------------------------------------------------------------
# subtitle / data stream types + src refs
# ---------------------------------------------------------------------------


def test_src_parts_subtitle() -> None:
    assert src_parts("src:a:s:0") == ("a", "subtitle", 0)


def test_src_parts_data() -> None:
    assert src_parts("src:a:d:3") == ("a", "data", 3)


def test_output_round_trip_subtitle() -> None:
    o = Output(
        ref="src:s:s:0",
        type="subtitle",
        name=None,
        metadata={"language": "eng"},
    )
    d = o.to_dict()
    o2 = Output.from_dict(d)
    assert o2 == o
    assert d["type"] == "subtitle"


def test_output_round_trip_data() -> None:
    o = Output(ref="src:a:d:0", type="data", name=None, metadata={})
    d = o.to_dict()
    o2 = Output.from_dict(d)
    assert o2 == o
    assert d["type"] == "data"


def test_output_from_dict_rejects_invalid_stream_type() -> None:
    with pytest.raises(ValueError):
        Output.from_dict(
            {"ref": "src:a:x:0", "type": "bogus", "name": None, "metadata": {}}
        )


def test_node_outputs_reject_subtitle_data_type_strings_are_still_parseable() -> None:
    # Node.outputs is typed StreamType too -- subtitle/data parse, even though
    # by the passthrough-only rule no node ever legitimately declares
    # them (that constraint is enforced by later passes, not by ir.py).
    n = Node(
        id="n0",
        filter="noop",
        args={},
        inputs=["src:a:s:0"],
        outputs=["subtitle"],
    )
    d = n.to_dict()
    n2 = Node.from_dict(d)
    assert n2.outputs == ["subtitle"]


# ---------------------------------------------------------------------------
# Graph.input_trims and Graph.input_options: additive, omitted when empty
# ---------------------------------------------------------------------------


def test_the_additive_input_fields_default_empty_and_stay_out_of_the_dict() -> None:
    g = _build_graph()
    d = g.to_dict()
    assert g.input_trims == {}
    assert g.input_options == {}
    assert "input_trims" not in d
    assert "input_options" not in d
    # from_dict reads an older payload that has neither key
    g2 = Graph.from_dict(d)
    assert g2.input_trims == {}
    assert g2.input_options == {}


def test_every_input_trim_shape_round_trips() -> None:
    g = _build_graph()
    # a closed window on one alias, an open UPPER bound on another
    g.input_trims = {"a": (5.0, 60.0), "b": (2.5, None)}
    d1 = g.to_dict()
    assert d1["input_trims"] == {"a": [5.0, 60.0], "b": [2.5, None]}
    g2 = Graph.from_dict(d1)
    assert g2.input_trims == {"a": (5.0, 60.0), "b": (2.5, None)}
    assert g2.to_dict() == d1
    # an open LOWER bound is the same JSON null in the other slot
    g.input_trims = {"a": (None, 60.0)}
    d2 = g.to_dict()
    assert d2["input_trims"] == {"a": [None, 60.0]}
    g3 = Graph.from_dict(d2)
    assert g3.input_trims == {"a": (None, 60.0)}
    assert g3.to_dict() == d2


def test_graph_input_options_round_trip() -> None:
    g = _build_graph()
    options: dict[str, dict[str, object]] = {
        "a": {"loop": True, "framerate": 15},
        "b": {"hwaccel": "cuda"},
    }
    g.input_options = options
    d1 = g.to_dict()
    assert d1["input_options"] == options
    g2 = Graph.from_dict(d1)
    assert g2.input_options == options
    assert g2.to_dict() == d1


def test_graph_input_options_are_independent_of_sinks_and_input_trims() -> None:
    """The two "additive, omit-when-empty" fields do not interfere."""
    g = _build_graph()
    g.input_options = {"a": {"itsoffset": -1.5}}
    g.input_trims = {"a": (0.0, 2.0)}
    d = g.to_dict()
    assert d["input_options"] == {"a": {"itsoffset": -1.5}}
    assert d["input_trims"] == {"a": [0.0, 2.0]}
    assert _serialized_sinks(d)[0]["path"] is None
    g2 = Graph.from_dict(d)
    assert g2.input_options == g.input_options
    assert g2.input_trims == g.input_trims
