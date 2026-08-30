"""End-to-end exec tests for ``LANGUAGE wasm``: a real module, really run.

Marked ``@pytest.mark.exec`` and excluded from the default run, same as
``test_exec.py``. Run explicitly::

    python -m pytest -m exec tests/exec/test_exec_wasm.py -q

Everything here is real: the query is compiled against the installed ffmpeg
and the sidecar's own ``--describe``, and ``execute_plan`` spawns the whole
pipeline -- an ffmpeg that decodes, ``ffrwd-wasm`` hosting the module, and an
ffmpeg that encodes. Requires ``ffmpeg``/``ffprobe`` on PATH, the generated
fixtures, the ``ffrwd-wasm`` sidecar (``uv sync --extra wasm``) and the
``invert`` module from the sidecar's vendored fleet, built for
``wasm32-wasip2``. Tests skip cleanly when any of those is missing.

The output is written with ``ffv1``, which is lossless, so what comes back
can be compared to the source BYTE FOR BYTE -- the same assertion the
sidecar's own ffmpeg test makes, one layer up: every R, G and B byte is its
source's complement and alpha is untouched.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ffrwd import binaries, wasm
from ffrwd.compiler import Compiled, compile_all, compile_sql
from ffrwd.emit import build_ffmpeg_args, emit
from ffrwd.errors import FfrwdError
from ffrwd.execute import execute_plan
from ffrwd.processes import PIPE
from ffrwd.project import LOCK_FORMAT_VERSION, discover

pytestmark = pytest.mark.exec

_CLI_ROOT = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _CLI_ROOT.parent
_FIXTURES_DIR = _CLI_ROOT / "tests" / "fixtures"
_SOURCE = _FIXTURES_DIR / "testsrc.mp4"


def _built(workspace: Path, name: str) -> Path:
    """A module's `.wasm`, in the workspace that builds it."""
    return workspace / "target" / "wasm32-wasip2" / "release" / f"{name}.wasm"


# One workspace builds every module named here: the sidecar's vendored fleet.
_SIDECAR_MODULES = _REPO_ROOT / "sidecar" / "modules"

_MODULE = _built(_SIDECAR_MODULES, "invert")
_SUBPROCESS_TIMEOUT = 120.0


@pytest.fixture(autouse=True)
def _require_everything() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found on PATH")
    if not _SOURCE.exists():
        pytest.skip(f"fixture missing: {_SOURCE} (run scripts/gen_fixtures.py first)")
    if binaries.ffrwd_wasm_path() is None:
        pytest.skip("ffrwd-wasm not found (uv sync --extra wasm)")
    if not _MODULE.exists():
        pytest.skip(
            f"module missing: {_MODULE} (cargo build --target wasm32-wasip2 "
            f"--release, from {_SIDECAR_MODULES})"
        )


def _query(out_path: Path) -> str:
    return (
        "CREATE FUNCTION invert(v video_stream) RETURNS video_stream\n"
        f"  AS '{_MODULE.as_posix()}', 'invert' LANGUAGE wasm;\n"
        "COPY (\n"
        "  SELECT invert(f.video[1])\n"
        f"  FROM input('{_SOURCE.as_posix()}') f\n"
        f") TO '{out_path.as_posix()}' WITH (video_codec 'ffv1')"
    )


def _rgba(path: Path) -> bytes:
    """Every frame of `path` decoded to packed RGBA."""
    done = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgba", "-",
        ],
        capture_output=True,
        timeout=_SUBPROCESS_TIMEOUT,
        check=False,
    )
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    return done.stdout


def _video_stream(path: Path) -> dict[str, object]:
    done = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames,width,height",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    streams = json.loads(done.stdout)["streams"]
    assert streams, f"{path} carries no video stream"
    result: dict[str, object] = streams[0]
    return result


@pytest.fixture(scope="module")
def _inverted(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The fixture's video, run through the module, written once for every test."""
    if shutil.which("ffmpeg") is None or binaries.ffrwd_wasm_path() is None:
        pytest.skip("ffmpeg or ffrwd-wasm missing")
    if not _SOURCE.exists() or not _MODULE.exists():
        pytest.skip("fixture or module missing")
    out_path = tmp_path_factory.mktemp("wasm") / "inverted.mkv"
    compiled = compile_all(_query(out_path))
    assert compiled.plan is not None, "a query naming a module compiles to a plan"

    result = execute_plan(
        compiled.plan,
        sidecar_argv=wasm.sidecar_argv,
        overwrite=True,
        timeout=_SUBPROCESS_TIMEOUT,
    )
    assert result.exit_code == 0, "\n".join(
        f"{member.id} exited {member.exit_code}: {member.stderr_tail}"
        for stage in result.stages
        for member in stage.members
    )
    assert not result.timed_out
    assert out_path.exists()
    return out_path


def test_the_pipeline_is_three_processes(tmp_path: Path) -> None:
    compiled = compile_all(_query(tmp_path / "out.mkv"))
    assert compiled.plan is not None
    assert len(compiled.plan.ffmpeg) == 2
    assert len(compiled.plan.sidecars) == 1


def test_the_real_module_describes_a_world_this_ffrwd_hosts(tmp_path: Path) -> None:
    described = wasm.describe(str(_MODULE))
    assert described.world in wasm.WORLDS
    assert described.name == "invert"
    assert wasm.wire_pix_fmt(described) in wasm.WIRE_PIX_FMTS


def test_the_output_keeps_every_frame(_inverted: Path) -> None:
    source = _video_stream(_SOURCE)
    inverted = _video_stream(_inverted)
    assert inverted["nb_read_frames"] == source["nb_read_frames"]
    assert (inverted["width"], inverted["height"]) == (
        source["width"],
        source["height"],
    )


def test_every_colour_byte_is_its_sources_complement(_inverted: Path) -> None:
    """The module really ran: R, G and B come back as 255 minus the source.

    The same check the sidecar's own ffmpeg test makes on raw frames, made
    here on the FILE the query wrote -- so it covers the whole pipeline, the
    two ffmpeg seams included.
    """
    source = _rgba(_SOURCE)
    inverted = _rgba(_inverted)
    assert len(inverted) == len(source)
    assert source, "the source decoded to nothing"

    wrong = [
        (index, source[index], inverted[index])
        for index in range(len(source))
        if index % 4 != 3 and inverted[index] != 255 - source[index]
    ]
    assert not wrong, f"{len(wrong)} colour bytes were not inverted, e.g. {wrong[:4]}"


def test_alpha_rides_through_untouched(_inverted: Path) -> None:
    source = _rgba(_SOURCE)
    inverted = _rgba(_inverted)
    assert source[3::4] == inverted[3::4]


def test_the_output_is_not_the_source(_inverted: Path) -> None:
    """The complement check would pass on an all-128 picture; this says it did not."""
    assert _rgba(_SOURCE) != _rgba(_inverted)


# -- one stream read by two processes: the in-graph half and the module ----


def _twin_query(out_path: Path) -> str:
    return (
        "CREATE FUNCTION invert(v video_stream) RETURNS video_stream\n"
        f"  AS '{_MODULE.as_posix()}', 'invert' LANGUAGE wasm;\n"
        "COPY (\n"
        "  SELECT ffmpeg.hstack(s.video[1], invert(s.video[1]))\n"
        f"  FROM input('{_SOURCE.as_posix()}') s\n"
        f") TO '{out_path.as_posix()}' WITH (video_codec 'libx264')"
    )


def test_a_stream_split_across_processes_leaves_no_pad_unconnected(
    tmp_path: Path,
) -> None:
    """One ref feeds hstack here and the module there: the producer is
    duplicated, and neither duplicate keeps a split it cannot connect."""
    compiled = compile_all(_twin_query(tmp_path / "out.mp4"))
    assert compiled.plan is not None
    assert len(compiled.plan.ffmpeg) == 3
    assert len(compiled.plan.sidecars) == 1

    producers = [
        p
        for p in compiled.plan.ffmpeg
        if any(unit.path == PIPE for unit in p.graph.sinks)
    ]
    assert len(producers) == 2
    for producer in producers:
        assert producer.graph.nodes == {}


def test_the_twinned_query_runs_to_a_playable_file(tmp_path: Path) -> None:
    """The reproduction, run: two decodes, a module and an hstack, one file."""
    out_path = tmp_path / "twin.mp4"
    compiled = compile_all(_twin_query(out_path))
    assert compiled.plan is not None

    result = execute_plan(
        compiled.plan,
        sidecar_argv=wasm.sidecar_argv,
        overwrite=True,
        timeout=_SUBPROCESS_TIMEOUT,
    )
    assert result.exit_code == 0, "\n".join(
        f"{member.id} exited {member.exit_code}: {member.stderr_tail}"
        for stage in result.stages
        for member in stage.members
    )
    assert not result.timed_out

    source = _video_stream(_SOURCE)
    twin = _video_stream(out_path)
    assert twin["width"] == int(str(source["width"])) * 2
    assert twin["height"] == source["height"]
    assert twin["nb_read_frames"] == source["nb_read_frames"]


# -- the annotated chain: one module's rows reaching the next --------------

_FACEBOX = _built(_SIDECAR_MODULES, "facebox")
_BLUR_BOXES = _built(_SIDECAR_MODULES, "blur_boxes")
_AV = _FIXTURES_DIR / "av.mp4"
_BOX = "STRUCT(x number, y number, w number, h number)[]"


def _require_annotation_modules() -> None:
    for module in (_FACEBOX, _BLUR_BOXES):
        if not module.exists():
            pytest.skip(f"module missing: {module}")
    if not _AV.exists():
        pytest.skip(f"fixture missing: {_AV}")


def _blur_query(out_path: Path) -> str:
    return (
        "CREATE FUNCTION detect_faces(v video_stream)\n"
        f"RETURNS STRUCT(v video_stream, faces {_BOX})\n"
        f"  AS '{_FACEBOX.as_posix()}', 'facebox' LANGUAGE wasm;\n"
        f"CREATE FUNCTION blur_boxes(v video_stream, faces {_BOX})\n"
        "RETURNS video_stream\n"
        f"  AS '{_BLUR_BOXES.as_posix()}', 'blur-boxes' LANGUAGE wasm;\n"
        "COPY (\n"
        "  SELECT blur_boxes(detect_faces(s.video[1])), s.audio\n"
        f"  FROM input('{_AV.as_posix()}') s\n"
        f") TO '{out_path.as_posix()}' WITH (video_codec 'ffv1')"
    )


def test_the_annotated_chain_is_three_processes(tmp_path: Path) -> None:
    """The two modules are adjacent, so one sidecar hosts both."""
    _require_annotation_modules()
    compiled = compile_all(_blur_query(tmp_path / "out.mkv"))
    assert compiled.plan is not None
    assert len(compiled.plan.ffmpeg) == 2
    assert len(compiled.plan.sidecars) == 1
    assert [b.path for b in compiled.plan.sidecars[0].modules] == [
        _FACEBOX.as_posix(),
        _BLUR_BOXES.as_posix(),
    ]


def test_the_rows_between_the_two_modules_never_reach_an_edge(tmp_path: Path) -> None:
    _require_annotation_modules()
    compiled = compile_all(_blur_query(tmp_path / "out.mkv"))
    assert compiled.plan is not None
    assert not any(e.annotations for e in compiled.plan.stream_edges)


def test_the_hosting_sidecar_is_run_as_a_network(tmp_path: Path) -> None:
    _require_annotation_modules()
    compiled = compile_all(_blur_query(tmp_path / "out.mkv"))
    assert compiled.plan is not None
    argv = wasm.shown_argv(compiled.plan.sidecars[0])
    assert "-annotations" not in argv
    assert argv.count("-m") == 2
    assert argv[argv.index("-m") + 1] == f"facebox={_FACEBOX.as_posix()}"
    network = argv[argv.index("-filter_complex") + 1]
    assert network == "[0:v]facebox[n1];[n1]blur_boxes[out0]"


def test_the_real_modules_describe_what_the_declarations_say(tmp_path: Path) -> None:
    _require_annotation_modules()
    facebox = wasm.describe(str(_FACEBOX))
    blur = wasm.describe(str(_BLUR_BOXES))
    assert facebox.world in wasm.WORLDS and blur.world in wasm.WORLDS
    assert (facebox.name, blur.name) == ("facebox", "blur-boxes")
    assert wasm.rows_fields(facebox) == (
        ("h", "integer"), ("w", "integer"), ("x", "integer"), ("y", "integer")
    )


@pytest.fixture(scope="module")
def _blurred(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The fixture run through the pair, written once for every test below.

    The fixture carries no faces, so nothing here asserts a blurred pixel:
    what is under test is that the rows one module writes reach the next, and
    that both seams survive it. Proving the blur itself needs real faces.
    """
    if shutil.which("ffmpeg") is None or binaries.ffrwd_wasm_path() is None:
        pytest.skip("ffmpeg or ffrwd-wasm missing")
    _require_annotation_modules()
    out_path = tmp_path_factory.mktemp("annotated") / "blurred.mkv"
    compiled = compile_all(_blur_query(out_path))
    assert compiled.plan is not None
    result = execute_plan(
        compiled.plan,
        sidecar_argv=wasm.sidecar_argv,
        overwrite=True,
        timeout=_SUBPROCESS_TIMEOUT,
    )
    assert result.exit_code == 0, "\n".join(
        f"{member.id} exited {member.exit_code}: {member.stderr_tail}"
        for stage in result.stages
        for member in stage.members
    )
    assert not result.timed_out
    assert out_path.exists()
    return out_path


def test_the_annotated_chain_keeps_every_frame(_blurred: Path) -> None:
    source = _video_stream(_AV)
    blurred = _video_stream(_blurred)
    assert blurred["nb_read_frames"] == source["nb_read_frames"]
    assert (blurred["width"], blurred["height"]) == (
        source["width"],
        source["height"],
    )


# -- what a windowed module declares, read from the real sidecar -----------

_DOUBLE = _built(_SIDECAR_MODULES, "double")
_TAIL3 = _built(_SIDECAR_MODULES, "tail3")


def _windowed_query(module: Path, export: str) -> str:
    return (
        f"CREATE FUNCTION shaped(v video_stream) RETURNS video_stream\n"
        f"  AS '{module.as_posix()}', '{export}' LANGUAGE wasm;\n"
        f"COPY (SELECT shaped(f.video[1]) FROM input('{_SOURCE.as_posix()}') f)\n"
        "TO 'shaped.mkv'"
    )


def test_a_windowed_module_describes_its_shape() -> None:
    if not _TAIL3.exists():
        pytest.skip(f"module missing: {_TAIL3}")
    described = wasm.describe(str(_TAIL3))
    assert described.windowed
    assert (described.window, described.stride) == (3, 3)
    assert described.one_to_one


def test_a_windowed_modules_lookahead_reaches_the_plan() -> None:
    if not _TAIL3.exists():
        pytest.skip(f"module missing: {_TAIL3}")
    compiled = compile_all(_windowed_query(_TAIL3, "tail3"))
    assert compiled.plan is not None
    assert compiled.plan.sidecars[0].lookahead == 2


def test_a_module_that_declares_itself_not_one_to_one_is_still_declarable() -> None:
    """`meta` on a windowed module says rows ride its calls, not that it reads any."""
    if not _DOUBLE.exists():
        pytest.skip(f"module missing: {_DOUBLE}")
    described = wasm.describe(str(_DOUBLE))
    # The sidecar now says reads_rows outright; a windowed carrier that
    # does not consume reports False on both.
    assert described.reads_rows is False and not described.reads_annotations
    assert not described.one_to_one
    compiled = compile_all(_windowed_query(_DOUBLE, "double"))
    assert compiled.plan is not None
    assert compiled.plan.sidecars[0].lookahead == 0


# -- a value-returning wasm function: folded at compile time, run once ------

_BRAND = _built(_SIDECAR_MODULES, "brand")
_TAGGED = _FIXTURES_DIR / "tagged.mp4"
_RESTORED_TITLE = "Angel One (restored)"


def _require_brand_module() -> None:
    if not _BRAND.exists():
        pytest.skip(f"module missing: {_BRAND}")
    if not _TAGGED.exists():
        pytest.skip(f"fixture missing: {_TAGGED}")


def _brand_query(out_path: Path) -> str:
    return (
        "CREATE FUNCTION brand(title text, suffix text) RETURNS text\n"
        f"  AS '{_BRAND.as_posix()}', 'append-brand' LANGUAGE wasm;\n"
        "COPY (\n"
        "  SELECT f.video[1], f.audio[1],\n"
        "         f.tags || STRUCT(brand(f.tags.title, ' (restored)') AS title) AS tags\n"
        f"  FROM input('{_TAGGED.as_posix()}') f\n"
        f") TO '{out_path.as_posix()}'"
    )


def _format_tags(path: Path) -> dict[str, str]:
    done = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format_tags",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    tags = json.loads(done.stdout)["format"].get("tags", {})
    result: dict[str, str] = tags
    return result


def test_a_value_function_compiles_to_no_plan(tmp_path: Path) -> None:
    """The module ran already, at compile time: nothing is left for the
    sidecar to host, so the query stays one plain ffmpeg command."""
    _require_brand_module()
    compiled = compile_all(_brand_query(tmp_path / "out.mp4"))
    assert compiled.plan is None
    assert compiled.graphs[0].sinks[0].tags["title"] == _RESTORED_TITLE


def test_the_folded_value_is_a_literal_in_the_argv(tmp_path: Path) -> None:
    out_path = tmp_path / "out.mp4"
    _require_brand_module()
    graph = compile_sql(_brand_query(out_path))
    args = build_ffmpeg_args(emit(graph), str(out_path))
    assert f"title={_RESTORED_TITLE}" in args
    assert not any("brand" in arg for arg in args)


def test_a_real_run_writes_the_folded_title(tmp_path: Path) -> None:
    """Compile, run the single ffmpeg command for real, and read the tag back."""
    _require_brand_module()
    out_path = tmp_path / "restored.mp4"
    graph = compile_sql(_brand_query(out_path))
    args = build_ffmpeg_args(emit(graph), str(out_path))
    args.insert(1, "-y")
    result = subprocess.run(
        args, capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT
    )
    assert result.returncode == 0, result.stderr
    assert out_path.exists()
    assert _format_tags(out_path)["title"] == _RESTORED_TITLE


def test_a_module_that_rejects_the_call_is_a_typed_error(tmp_path: Path) -> None:
    """``append-brand`` requires both `title` and `suffix`; a declaration
    that only forwards one omits the other from the call, and the module's
    own stderr -- not a generic failure -- is what the rejection carries."""
    _require_brand_module()
    sql = (
        "CREATE FUNCTION half_brand(title text) RETURNS text\n"
        f"  AS '{_BRAND.as_posix()}', 'append-brand' LANGUAGE wasm;\n"
        "COPY (SELECT f.video[1], STRUCT(half_brand('X') AS title) AS tags\n"
        f"      FROM input('{_TAGGED.as_posix()}') f) "
        f"TO '{(tmp_path / 'out.mp4').as_posix()}'"
    )
    with pytest.raises(FfrwdError) as caught:
        compile_all(sql)
    error = caught.value
    assert "half_brand" in error.message
    assert "suffix" in error.message
    assert error.line is not None


# -- a module shipped inside a linked package ------------------------------
#
# The module path is written relative to the package that declares it, and the
# query is compiled from a working directory that is neither the package's nor
# the project's -- so a path read against the cwd would not find the file at
# all. The package is LINKED rather than installed, which needs no registry
# and no store.


def _linked_project(root: Path) -> Path:
    """A project linking a package that ships `invert.wasm` and declares it.

    Returns the project directory. The module is copied in, so the resolved
    path is the package's own copy and not the built one this file names.
    """
    package = root / "tools"
    (package / "src").mkdir(parents=True)
    (package / "modules").mkdir(parents=True)
    shutil.copy(_MODULE, package / "modules" / "invert.wasm")
    (package / "src" / "tools.sql").write_text(
        "CREATE FUNCTION invert(v video_stream) RETURNS video_stream\n"
        "  AS 'modules/invert.wasm', 'invert' LANGUAGE wasm;\n",
        encoding="utf-8",
    )
    (package / "ffrwd.json").write_text(
        json.dumps(
            {
                "name": "ffrwd/tools",
                "version": "1.0.0",
                "lib": {"invert": "src/tools.sql"},
            }
        ),
        encoding="utf-8",
    )
    project = root / "work"
    project.mkdir()
    (project / "ffrwd.json").write_text(
        json.dumps({"name": "me/work", "version": "0.1.0"}), encoding="utf-8"
    )
    (project / "ffrwd.lock").write_text(
        json.dumps(
            {
                "format_version": LOCK_FORMAT_VERSION,
                "reproducible": False,
                "not_reproducible_because": "a package is linked",
                "packages": [{"kind": "link", "path": "../tools"}],
            }
        ),
        encoding="utf-8",
    )
    return project


def _package_query(out_path: Path) -> str:
    return (
        "COPY (\n"
        "  SELECT ffrwd.tools.invert(f.video[1])\n"
        f"  FROM input('{_SOURCE.as_posix()}') f\n"
        f") TO '{out_path.as_posix()}' WITH (video_codec 'ffv1')"
    )


def _compiled_from_elsewhere(
    root: Path, out_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Compiled:
    project = _linked_project(root)
    packages = discover(project)
    assert packages is not None
    elsewhere = root / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    return compile_all(_package_query(out_path), packages=packages)


def test_a_linked_packages_module_resolves_against_the_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiled = _compiled_from_elsewhere(tmp_path, tmp_path / "out.mkv", monkeypatch)
    plan = compiled.plan
    assert plan is not None
    assert [Path(s.module) for s in plan.sidecars] == [
        tmp_path / "tools" / "modules" / "invert.wasm"
    ]


def test_a_linked_packages_module_really_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole pipeline, from a cwd where the written path names nothing."""
    out_path = tmp_path / "out.mkv"
    compiled = _compiled_from_elsewhere(tmp_path, out_path, monkeypatch)
    plan = compiled.plan
    assert plan is not None
    result = execute_plan(
        plan,
        sidecar_argv=wasm.sidecar_argv,
        overwrite=True,
        timeout=_SUBPROCESS_TIMEOUT,
    )
    assert result.exit_code == 0, "\n".join(
        f"{member.id} exited {member.exit_code}: {member.stderr_tail}"
        for stage in result.stages
        for member in stage.members
    )
    assert out_path.exists()
    source, inverted = _rgba(_SOURCE), _rgba(out_path)
    assert len(inverted) == len(source) and source
    wrong = [
        index
        for index in range(len(source))
        if index % 4 != 3 and inverted[index] != 255 - source[index]
    ]
    assert not wrong, f"{len(wrong)} colour bytes were not inverted"



# -- sink modules: the graph ends inside the sidecar -----------------------

_FRAME_STATS = _built(_SIDECAR_MODULES, "frame_stats")
_POST_ROWS = _built(_SIDECAR_MODULES, "post_rows")
_FRAMESTATS = _built(_SIDECAR_MODULES, "framestats")


def _require_sink_modules() -> None:
    for module in (_FRAME_STATS, _POST_ROWS, _FRAMESTATS):
        if not module.exists():
            pytest.skip(
                f"module missing: {module} (cargo build --target wasm32-wasip2 "
                f"--release, from {_SIDECAR_MODULES})"
            )


def _run_plan(sql: str) -> object:
    compiled = compile_all(sql)
    assert compiled.plan is not None
    result = execute_plan(
        compiled.plan,
        sidecar_argv=wasm.sidecar_argv,
        overwrite=True,
        timeout=_SUBPROCESS_TIMEOUT,
    )
    assert result.exit_code == 0, "\n".join(
        f"{member.id} exited {member.exit_code}: {member.stderr_tail}"
        for stage in result.stages
        for member in stage.members
    )
    assert not result.timed_out
    return result


def test_a_frame_sink_consumes_the_stream_and_reports_on_stderr() -> None:
    """The pipeline completes when the input drains, and the sink saw the end."""
    _require_sink_modules()
    sql = (
        "CREATE FUNCTION frame_stats(v video_stream) RETURNS sink\n"
        f"  AS '{_FRAME_STATS.as_posix()}', 'frame_stats' LANGUAGE wasm;\n"
        "COPY (\n"
        "  SELECT f.video[1]\n"
        f"  FROM input('{_SOURCE.as_posix()}') f\n"
        ") TO frame_stats()"
    )
    result = _run_plan(sql)
    tails = " ".join(
        member.stderr_tail or ""
        for stage in result.stages  # type: ignore[attr-defined]
        for member in stage.members
    )
    frames = _video_stream(_SOURCE)["nb_read_frames"]
    # stderr_tail keeps the END of the stream of lines: the closing count is
    # there, and per-frame lines led up to it.
    assert "frame_stats frame=" in tails
    assert f"frame_stats frames={frames}" in tails


def test_a_rows_sink_posts_every_row_to_the_endpoint() -> None:
    """Rows cross the module seam, the grant, and a real HTTP hop."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    _require_sink_modules()
    received: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("content-length", 0))
            received.append(self.rfile.read(length).decode("utf-8"))
            self.send_response(200)
            self.send_header("content-length", "0")
            self.end_headers()

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        sql = (
            "CREATE FUNCTION stats(v video_stream)\n"
            "RETURNS STRUCT(v video_stream, samples STRUCT(time number, mean number)[])\n"
            f"  AS '{_FRAMESTATS.as_posix()}', 'framestats' LANGUAGE wasm;\n"
            "CREATE FUNCTION post_rows(v video_stream,\n"
            "                          samples STRUCT(time number, mean number)[],\n"
            "                          url text)\n"
            "RETURNS sink\n"
            f"  AS '{_POST_ROWS.as_posix()}', 'post_rows' LANGUAGE wasm;\n"
            "COPY (\n"
            "  SELECT stats(f.video[1])\n"
            f"  FROM input('{_SOURCE.as_posix()}') f\n"
            f") TO post_rows('http://127.0.0.1:{port}/rows')"
        )
        _run_plan(sql)
    finally:
        server.shutdown()
        thread.join(timeout=5)

    frames = _video_stream(_SOURCE)["nb_read_frames"]
    # One POST per frame's row, plus the one trailing summary row.
    assert len(received) == int(str(frames)) + 1
    rows = [json.loads(body) for body in received]
    assert all("time" in row and "mean" in row for row in rows[:-1])
    assert rows[-1] == {"frames": int(str(frames))}


def test_a_rows_sink_fails_the_run_when_the_endpoint_refuses() -> None:
    """A failed POST is a failed run, not a silent drop."""
    _require_sink_modules()
    sql = (
        "CREATE FUNCTION stats(v video_stream)\n"
        "RETURNS STRUCT(v video_stream, samples STRUCT(time number, mean number)[])\n"
        f"  AS '{_FRAMESTATS.as_posix()}', 'framestats' LANGUAGE wasm;\n"
        "CREATE FUNCTION post_rows(v video_stream,\n"
        "                          samples STRUCT(time number, mean number)[],\n"
        "                          url text)\n"
        "RETURNS sink\n"
        f"  AS '{_POST_ROWS.as_posix()}', 'post_rows' LANGUAGE wasm;\n"
        "COPY (\n"
        "  SELECT stats(f.video[1])\n"
        f"  FROM input('{_SOURCE.as_posix()}') f\n"
        ") TO post_rows('http://127.0.0.1:9/unreachable')"
    )
    compiled = compile_all(sql)
    assert compiled.plan is not None
    result = execute_plan(
        compiled.plan,
        sidecar_argv=wasm.sidecar_argv,
        overwrite=True,
        timeout=_SUBPROCESS_TIMEOUT,
    )
    assert result.exit_code != 0


# -- packet sinks: the encoded edge, really run ----------------------------

_PACKET_STATS = _built(_SIDECAR_MODULES, "packet_stats")


def test_a_packet_sink_reads_the_encoders_output_and_its_rows_arrive(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """The feeder encodes, the sidecar hands packets through, rows reach stdout.

    A lavfi source of 10 frames under ``gop 5``: every frame becomes one
    packet, so the trailing summary counts 10, and each closed group's row
    precedes it. libx264 may add keyframes of its own on a scene change, so
    the keyframe count is bounded below, not pinned.
    """
    if not _PACKET_STATS.exists():
        pytest.skip(
            f"module missing: {_PACKET_STATS} (cargo build --target "
            f"wasm32-wasip2 --release, from {_SIDECAR_MODULES})"
        )
    sql = (
        "CREATE FUNCTION packet_stats(v video_stream) RETURNS sink\n"
        f"  AS '{_PACKET_STATS.as_posix()}', 'packet_stats' LANGUAGE wasm;\n"
        "COPY (\n"
        "  SELECT s.video[1]\n"
        "  FROM input('testsrc2=size=64x64:rate=10:duration=1', format => 'lavfi') s\n"
        ") TO packet_stats() WITH (video_codec 'libx264', gop 5)"
    )
    _run_plan(sql)
    lines = [line for line in capfd.readouterr().out.splitlines() if line.strip()]
    rows = [json.loads(line) for line in lines]
    assert rows, "the sink's rows never reached stdout"
    summary = rows[-1]
    assert summary["packets"] == 10
    assert summary["keyframes"] >= 2
    assert summary["gops"] == summary["keyframes"]
    assert summary["bytes"] > 0
    # One row per closed group precedes the summary, each naming its group.
    assert len(rows) - 1 == summary["gops"]
    assert all(row["gop"] == index for index, row in enumerate(rows[:-1]))
    assert sum(row["packets"] for row in rows[:-1]) == 10


# --- a live-paced source, read once ------------------------------------------

# A lavfi graph is a one-open input like a camera or a listener: its `format
# =>` names the demuxer, so the compiler reads it with a single process. `-re`
# paces it at the source's own frame rate, which is what makes the two paths
# out of that process race each other for real.
_LIVE_FRAMES = 30
_LIVE_SIZE = (160, 120)


def _live_query(out_path: Path) -> str:
    width, height = _LIVE_SIZE
    return (
        "CREATE FUNCTION invert(v video_stream) RETURNS video_stream\n"
        f"  AS '{_MODULE.as_posix()}', 'invert' LANGUAGE wasm;\n"
        "COPY (\n"
        "  SELECT ffmpeg.hstack(a.video[1], invert(a.video[1]))\n"
        f"  FROM input('testsrc2=size={width}x{height}:rate=30:duration=1',\n"
        "             format => 'lavfi', realtime => true) a\n"
        f") TO '{out_path.as_posix()}' WITH (video_codec 'ffv1')"
    )


@pytest.fixture(scope="module")
def _live_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Compiled]:
    """A live-paced source through the module and back into its own picture."""
    if shutil.which("ffmpeg") is None or binaries.ffrwd_wasm_path() is None:
        pytest.skip("ffmpeg or ffrwd-wasm missing")
    if not _MODULE.exists():
        pytest.skip(f"module missing: {_MODULE}")
    out_path = tmp_path_factory.mktemp("live") / "live.mkv"
    compiled = compile_all(_live_query(out_path))
    assert compiled.plan is not None

    result = execute_plan(
        compiled.plan,
        sidecar_argv=wasm.sidecar_argv,
        overwrite=True,
        timeout=_SUBPROCESS_TIMEOUT,
    )
    assert result.exit_code == 0, "\n".join(
        f"{member.id} exited {member.exit_code}: {member.stderr_tail}"
        for stage in result.stages
        for member in stage.members
    )
    assert not result.timed_out
    assert result.overflow is None, str(result.overflow)
    return out_path, compiled


def test_a_live_source_is_opened_by_exactly_one_process(
    _live_run: tuple[Path, Compiled],
) -> None:
    """The plan that just ran: one -i over the lavfi graph, and only one."""
    _, compiled = _live_run
    assert compiled.plan is not None
    spec = f"testsrc2=size={_LIVE_SIZE[0]}x{_LIVE_SIZE[1]}:rate=30:duration=1"
    opening = [p.id for p in compiled.plan.ffmpeg if spec in p.graph.input_paths]

    assert opening == ["ffmpeg1"]
    # It hands one pipe to each consumer, which is what a second open would
    # otherwise have been.
    assert len([e for e in compiled.plan.stream_edges if e.source == "ffmpeg1"]) == 2


def test_the_edge_that_waits_for_the_module_carries_its_bound(
    _live_run: tuple[Path, Compiled],
) -> None:
    _, compiled = _live_run
    assert compiled.plan is not None
    direct = next(
        e
        for e in compiled.plan.stream_edges
        if (e.source, e.target) == ("ffmpeg1", "ffmpeg0")
    )

    # One process stands between the reader and the merge, and `invert`
    # declares no window, so the direct edge holds exactly one frame.
    assert direct.bound == 1
    assert direct.buffer is not None
    assert (direct.buffer.road, direct.buffer.frames) == ("pipe", 2)


def test_the_live_run_wrote_every_frame_side_by_side(
    _live_run: tuple[Path, Compiled],
) -> None:
    """It completed rather than wedging, and both legs reached every frame."""
    out_path, _ = _live_run
    written = _video_stream(out_path)

    assert int(str(written["nb_read_frames"])) == _LIVE_FRAMES
    assert (written["width"], written["height"]) == (_LIVE_SIZE[0] * 2, _LIVE_SIZE[1])
