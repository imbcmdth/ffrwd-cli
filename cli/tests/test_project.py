"""Tests for projects and packages: the two files, discovery, and resolution.

Every project, lockfile and store is built under ``tmp_path``: nothing here
reads the working directory, the home directory or the network, and the filter
surface is the captured snapshot (tests/conftest.py), so ``compile_sql``
resolves ``volume`` on a machine with no ffmpeg. The store lives wherever
``store._cache_dir`` says, which the suite points at a temporary directory and
the ``store_home`` fixture points at one per test.

The headline check is :func:`test_a_package_call_compiles_to_the_inline_argv`:
a call into a package produces the same ffmpeg argv, byte for byte, as the
same body written into the query itself.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import subprocess
import tarfile
from collections.abc import Callable
from pathlib import Path

import pytest

from ffrwd import cli, store, wasm
from ffrwd.compiler import compile_commands, compile_sql, compile_table_sql
from ffrwd.emit import build_ffmpeg_args, emit
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.functions import package_signatures
from ffrwd.mcp import tools as mcp_tools
from ffrwd.project import (
    LOCK_FORMAT_VERSION,
    MANIFEST_NAME,
    LinkEntry,
    ModelPin,
    PackageSet,
    RegistryEntry,
    discover,
    find_manifest,
    read_lockfile,
    read_manifest,
    with_entry,
    without_entry,
    write_lockfile,
    write_manifest,
)
from ffrwd.warnings import FfrwdWarning, WarningCode

QUIETER = (
    "CREATE FUNCTION quieter(track audio_stream, factor number) RETURNS audio_stream AS $$\n"
    "  SELECT volume(track, factor)\n"
    "$$ LANGUAGE sql;\n"
)
PICK = (
    "CREATE FUNCTION pick(path text) RETURNS TABLE(track audio_stream) AS $$\n"
    "  SELECT f.audio[1] FROM input(path) f\n"
    "$$ LANGUAGE sql;\n"
)
NORMALIZE = (
    "CREATE FUNCTION normalize_lang(raw text) RETURNS text AS $$\n"
    "  SELECT CASE WHEN raw = 'english' THEN 'eng' ELSE raw END\n"
    "$$ LANGUAGE sql;\n"
)


def _derived_lib(files: dict[str, str]) -> dict[str, str]:
    """A map `lib` exporting every definition the src files hold, in file order."""
    lib: dict[str, str] = {}
    for name, body in files.items():
        if not name.startswith("src/"):
            continue
        for defined in re.findall(r"CREATE FUNCTION (\w+)\(", body):
            lib[defined] = name
    return lib


def _project(
    root: Path,
    *,
    files: dict[str, str] | None = None,
    manifest: dict[str, object] | None = None,
    text: str | None = None,
) -> Path:
    """Write a project under `root` and return its manifest path.

    The default manifest names the package ``me/edits`` and exports every
    definition the ``src/`` files hold; `manifest` overrides keys, `text`
    writes the file verbatim, for the malformed cases a dict cannot express.
    """
    for name, body in (files or {}).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    written = root / "ffrwd.json"
    written.parent.mkdir(parents=True, exist_ok=True)
    if text is not None:
        written.write_text(text, encoding="utf-8")
    else:
        declared: dict[str, object] = {"name": "me/edits", "version": "0.1.0"}
        lib = _derived_lib(files or {})
        if lib:
            declared["lib"] = lib
        declared.update(manifest or {})
        declared = {key: value for key, value in declared.items() if value is not None}
        written.write_text(json.dumps(declared, indent=2) + "\n", encoding="utf-8")
    return written


def _packages(root: Path) -> PackageSet:
    found = discover(root)
    assert found is not None
    return found


def _argv(sql: str, packages: PackageSet | None = None) -> list[str]:
    return build_ffmpeg_args(emit(compile_sql(sql, packages=packages)))


def _rejects(sql: str, packages: PackageSet | None, code: ErrorCode, needle: str) -> FfrwdError:
    with pytest.raises(FfrwdError) as caught:
        compile_commands(sql, packages=packages)
    error = caught.value
    assert error.code is code, f"{error.code} != {code}: {error}"
    assert needle in error.message, error.message
    return error


# ---------------------------------------------------------------------------
# the headline: a package call and the same body inline are one command
# ---------------------------------------------------------------------------


def test_a_package_call_compiles_to_the_inline_argv(tmp_path: Path) -> None:
    _project(tmp_path, files={"src/tracks.sql": QUIETER})
    query = (
        "COPY (SELECT f.video[1], {call}(f.audio[1], 0.5) FROM input('film.mkv') f) TO 'out.mkv'"
    )
    packaged = _argv(query.format(call="me.edits.quieter"), _packages(tmp_path))
    inline = _argv(QUIETER + query.format(call="quieter"))
    assert packaged == inline
    assert "[0:a:0]volume=volume=0.5[out1]" in " ".join(packaged)


def test_a_table_returning_package_function_is_a_row_source(tmp_path: Path) -> None:
    _project(tmp_path, files={"src/tracks.sql": PICK})
    query = "COPY (SELECT t.track FROM {call}('a.mka') t) TO 'out.mka'"
    packaged = _argv(query.format(call="me.edits.pick"), _packages(tmp_path))
    inline = _argv(PICK + query.format(call="pick"))
    assert packaged == inline
    assert packaged[:3] == ["ffmpeg", "-i", "a.mka"]


def test_a_package_function_reads_rows_as_a_table_query(tmp_path: Path) -> None:
    _project(tmp_path, files={"src/lang.sql": NORMALIZE})
    sinks = compile_table_sql(
        "SELECT me.edits.normalize_lang(said.raw) AS language "
        "FROM unnest(ARRAY[STRUCT('english' AS raw), STRUCT('de' AS raw)]) said",
        packages=_packages(tmp_path),
    )
    assert sinks[0].result.rows == [["eng"], ["de"]]


def test_the_default_lib_is_reached_as_the_package_segment(tmp_path: Path) -> None:
    """`lib`'s export is named for the package segment: `me.edits(...)`."""
    edits = (
        "CREATE FUNCTION edits(track audio_stream) RETURNS audio_stream AS $$\n"
        "  SELECT volume(track, 0.5)\n"
        "$$ LANGUAGE sql;\n"
    )
    _project(
        tmp_path,
        files={"src/default.sql": edits},
        manifest={"lib": "src/default.sql"},
    )
    argv = _argv(
        "COPY (SELECT me.edits(f.audio[1]) FROM input('film.mkv') f) TO 'out.mkv'",
        _packages(tmp_path),
    )
    assert "volume=volume=0.5" in " ".join(argv)


def test_a_dependency_is_called_by_its_full_path_never_an_alias(tmp_path: Path) -> None:
    """`broadcast.tracks.quieter(...)` -- the explicit path is the only path."""
    entry = _installed(_library(tmp_path / "built", "broadcast", "0.5", package="tracks"))
    project = tmp_path / "work"
    _project(
        project,
        files={"src/own.sql": NORMALIZE},
        manifest={"dependencies": {"broadcast/tracks": "^1.0.0"}},
    )
    _lock(project, [entry])
    packages = _packages(project)
    full = _argv(QUERY.format(call="broadcast.tracks.quieter"), packages)
    assert "volume=volume=0.5" in " ".join(full)


def test_a_one_segment_qualifier_is_not_a_package_lookup(tmp_path: Path) -> None:
    """What used to be an alias call is `UNKNOWN_FUNCTION`, hinting at the explicit form."""
    entry = _installed(_library(tmp_path / "built", "broadcast", "0.5", package="tracks"))
    project = tmp_path / "work"
    _project(
        project,
        files={"src/own.sql": NORMALIZE},
        manifest={"dependencies": {"broadcast/tracks": "^1.0.0"}},
    )
    _lock(project, [entry])
    error = _rejects(
        QUERY.format(call="tracks.quieter"),
        _packages(project),
        ErrorCode.UNKNOWN_FUNCTION,
        "unknown namespace 'tracks'",
    )
    assert error.hint == "calls across packages are written <namespace>.<package>.<member>"


def test_a_two_segment_call_on_a_map_lib_package_names_its_exports(
    tmp_path: Path,
) -> None:
    """`me.edits(...)` with a map `lib` has no root member; the exports are named instead."""
    _project(tmp_path, files={"src/tracks.sql": QUIETER})
    error = _rejects(
        "COPY (SELECT me.edits(f.audio[1], 0.5) FROM input('film.mkv') f) TO 'out.mkv'",
        _packages(tmp_path),
        ErrorCode.UNKNOWN_FUNCTION,
        "package 'me/edits' names its exports",
    )
    assert error.hint == "it exports: quieter"


def test_a_three_segment_call_reaches_a_package_in_the_official_namespace(
    tmp_path: Path,
) -> None:
    """`ffrwd/*` is claimable, and three segments reach it like any other package."""
    _project(tmp_path, files={"src/tracks.sql": QUIETER}, manifest={"name": "ffrwd/tools"})
    packaged = _argv(
        "COPY (SELECT ffrwd.tools.quieter(f.audio[1], 0.5) FROM input('film.mkv') f) "
        "TO 'out.mkv'",
        _packages(tmp_path),
    )
    assert "[0:a:0]volume=volume=0.5[out0]" in " ".join(packaged)


def test_a_macro_wins_a_two_segment_call_against_the_official_namespace(
    tmp_path: Path,
) -> None:
    """`ffrwd.<name>(...)` is two segments, so it is the macro whatever is installed."""
    _project(tmp_path, files={"src/tracks.sql": QUIETER}, manifest={"name": "ffrwd/tools"})
    argv = _argv(
        "COPY (SELECT ffrwd.speed(f.video[1], 2) FROM input('film.mkv') f) TO 'out.mkv'",
        _packages(tmp_path),
    )
    assert "[0:v:0]setpts=PTS/2[out0]" in " ".join(argv)


def test_a_two_segment_typo_under_the_official_namespace_still_suggests_a_macro(
    tmp_path: Path,
) -> None:
    """An installed `ffrwd/*` package does not take the macros' did-you-mean over."""
    _project(tmp_path, files={"src/tracks.sql": QUIETER}, manifest={"name": "ffrwd/tools"})
    error = _rejects(
        "COPY (SELECT ffrwd.spede(f.video[1], 2) FROM input('film.mkv') f) TO 'out.mkv'",
        _packages(tmp_path),
        ErrorCode.UNKNOWN_FUNCTION,
        "unknown function ffrwd.spede()",
    )
    assert error.hint == "did you mean ffrwd.speed()?"


def test_a_three_segment_typo_suggests_the_packages_own_exports(tmp_path: Path) -> None:
    _project(tmp_path, files={"src/tracks.sql": QUIETER}, manifest={"name": "ffrwd/tools"})
    error = _rejects(
        "COPY (SELECT ffrwd.tools.quiter(f.audio[1], 0.5) FROM input('film.mkv') f) "
        "TO 'out.mkv'",
        _packages(tmp_path),
        ErrorCode.UNKNOWN_FUNCTION,
        "package 'ffrwd/tools' has no export 'quiter'",
    )
    assert error.hint == "did you mean ffrwd.tools.quieter()?"


def test_a_two_segment_call_naming_an_official_package_says_to_write_it_in_full(
    tmp_path: Path,
) -> None:
    """The one form the two rules could be confused over, answered by name."""
    _project(tmp_path, files={"src/tracks.sql": QUIETER}, manifest={"name": "ffrwd/tools"})
    error = _rejects(
        "COPY (SELECT ffrwd.tools(f.audio[1], 0.5) FROM input('film.mkv') f) TO 'out.mkv'",
        _packages(tmp_path),
        ErrorCode.UNKNOWN_FUNCTION,
        "ffrwd.tools() is two segments, which is always a macro",
    )
    assert error.hint == "package 'ffrwd/tools' is called in full: ffrwd.tools.<member>(...)"


def test_a_two_segment_call_on_a_package_with_no_exports_at_all_says_so(
    tmp_path: Path,
) -> None:
    """No `lib` at all is worded differently from a map `lib` with nothing to default to."""
    _project(tmp_path, files={"queries/split.sql": RECIPE}, manifest=_BIN)
    error = _rejects(
        "COPY (SELECT me.edits(f.audio[1], 0.5) FROM input('film.mkv') f) TO 'out.mkv'",
        _packages(tmp_path),
        ErrorCode.UNKNOWN_FUNCTION,
        "package 'me/edits' has no default export",
    )
    assert error.hint == "me/edits exports nothing"


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def test_discovery_walks_up_from_a_subdirectory(tmp_path: Path) -> None:
    manifest = _project(tmp_path, files={"src/tracks.sql": QUIETER})
    deep = tmp_path / "queries" / "nested"
    deep.mkdir(parents=True)
    assert find_manifest(deep) == manifest
    found = discover(deep)
    assert found is not None
    assert found.names() == ("me/edits",)
    assert found.namespaces() == ("me",)


def test_discovery_accepts_a_query_file_path(tmp_path: Path) -> None:
    manifest = _project(tmp_path, files={"src/tracks.sql": QUIETER})
    query = tmp_path / "queries" / "out.sql"
    query.parent.mkdir()
    query.write_text("SELECT 1", encoding="utf-8")
    assert find_manifest(query) == manifest


def test_no_manifest_is_no_project(tmp_path: Path) -> None:
    bare = tmp_path / "no_project"
    bare.mkdir()
    assert find_manifest(bare) is None
    assert discover(bare) is None


def test_without_a_project_a_query_compiles_exactly_as_before(tmp_path: Path) -> None:
    bare = tmp_path / "no_project"
    bare.mkdir()
    sql = QUIETER + (
        "COPY (SELECT quieter(f.audio[1], 0.5) FROM input('film.mkv') f) TO 'out.mkv'"
    )
    assert _argv(sql, discover(bare)) == _argv(sql)


def test_without_a_project_a_namespaced_call_is_rejected_as_it_always_was(tmp_path: Path) -> None:
    bare = tmp_path / "no_project"
    bare.mkdir()
    sql = "COPY (SELECT me.edits.quieter(f.audio[1], 0.5) FROM input('film.mkv') f) TO 'out.mkv'"
    with pytest.raises(FfrwdError) as without_project:
        compile_commands(sql, packages=discover(bare))
    with pytest.raises(FfrwdError) as never_asked:
        compile_commands(sql)
    assert str(without_project.value) == str(never_asked.value)


# ---------------------------------------------------------------------------
# a package's lib files are a library; the script is not
# ---------------------------------------------------------------------------


def test_an_uncalled_package_definition_is_fine(tmp_path: Path) -> None:
    _project(tmp_path, files={"src/tracks.sql": QUIETER + NORMALIZE})
    argv = _argv(
        "COPY (SELECT me.edits.quieter(f.audio[1], 0.5) FROM input('film.mkv') f) TO 'out.mkv'",
        _packages(tmp_path),
    )
    assert argv[-1] == "out.mkv"


def test_an_uncalled_script_definition_is_still_an_error(tmp_path: Path) -> None:
    _project(tmp_path, files={"src/tracks.sql": QUIETER})
    _rejects(
        NORMALIZE + "COPY (SELECT f.video[1] FROM input('film.mkv') f) TO 'out.mkv'",
        _packages(tmp_path),
        ErrorCode.UNSUPPORTED_SQL,
        "'normalize_lang' is never called",
    )


def test_a_package_body_calls_its_own_sibling_not_the_script(tmp_path: Path) -> None:
    """A bare name in a library body means the library's own definition."""
    library = (
        "CREATE FUNCTION helper(track audio_stream) RETURNS audio_stream AS $$\n"
        "  SELECT volume(track, 0.25)\n"
        "$$ LANGUAGE sql;\n"
        "CREATE FUNCTION quieter(track audio_stream) RETURNS audio_stream AS $$\n"
        "  SELECT helper(track)\n"
        "$$ LANGUAGE sql;\n"
    )
    _project(tmp_path, files={"src/tracks.sql": library})
    shadow = (
        "CREATE FUNCTION helper(track audio_stream) RETURNS audio_stream AS $$\n"
        "  SELECT volume(track, 8)\n"
        "$$ LANGUAGE sql;\n"
    )
    argv = _argv(
        shadow + "COPY (SELECT me.edits.quieter(f.audio[1]), helper(f.audio[2]) "
        "FROM input('film.mkv') f) TO 'out.mkv'",
        _packages(tmp_path),
    )
    graph = " ".join(argv)
    assert "volume=volume=0.25" in graph
    assert "volume=volume=8" in graph


def test_a_definition_libs_does_not_name_is_private(tmp_path: Path) -> None:
    """A lib file may define more than the manifest exports; the rest are the package's own."""
    library = (
        "CREATE FUNCTION helper(track audio_stream) RETURNS audio_stream AS $$\n"
        "  SELECT volume(track, 0.25)\n"
        "$$ LANGUAGE sql;\n"
        "CREATE FUNCTION quieter(track audio_stream) RETURNS audio_stream AS $$\n"
        "  SELECT helper(track)\n"
        "$$ LANGUAGE sql;\n"
    )
    _project(
        tmp_path,
        files={"src/tracks.sql": library},
        manifest={"lib": {"quieter": "src/tracks.sql"}},
    )
    packages = _packages(tmp_path)
    argv = _argv(
        "COPY (SELECT me.edits.quieter(f.audio[1]) FROM input('film.mkv') f) TO 'out.mkv'", packages
    )
    assert "volume=volume=0.25" in " ".join(argv)
    error = _rejects(
        "COPY (SELECT me.edits.helper(f.audio[1]) FROM input('film.mkv') f) TO 'out.mkv'",
        packages,
        ErrorCode.UNKNOWN_FUNCTION,
        "package 'me/edits' has no export 'helper'",
    )
    assert error.hint == "me/edits exports: quieter"


# ---------------------------------------------------------------------------
# rejections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("claimed", ["ffmpeg", "wasm"])
def test_a_reserved_namespace_is_refused(tmp_path: Path, claimed: str) -> None:
    manifest = _project(
        tmp_path, files={"src/tracks.sql": QUIETER}, manifest={"name": f"{claimed}/edits"}
    )
    with pytest.raises(FfrwdError) as caught:
        read_manifest(manifest)
    assert f"namespace '{claimed}' is reserved" in caught.value.message
    assert caught.value.line == 2


def test_the_official_namespace_is_claimable(tmp_path: Path) -> None:
    """`ffrwd` is where the official packages are published, not a refusal."""
    manifest = _project(
        tmp_path, files={"src/tracks.sql": QUIETER}, manifest={"name": "ffrwd/tools"}
    )
    assert read_manifest(manifest).name == "ffrwd/tools"


@pytest.mark.parametrize("macro", ["speed", "delay", "empty_captions"])
def test_a_package_named_for_a_macro_is_refused(tmp_path: Path, macro: str) -> None:
    """`ffrwd.speed(...)` is two segments, so it is always the macro."""
    manifest = _project(
        tmp_path, files={"src/tracks.sql": QUIETER}, manifest={"name": f"ffrwd/{macro}"}
    )
    with pytest.raises(FfrwdError) as caught:
        read_manifest(manifest)
    assert f"is named for the macro ffrwd.{macro}()" in caught.value.message
    assert caught.value.hint is not None
    assert "rename the package" in caught.value.hint
    assert caught.value.line == 2


def test_an_unknown_namespace_says_what_this_project_has(tmp_path: Path) -> None:
    """The three-segment form: a genuinely unknown namespace, not a former alias."""
    _project(tmp_path, files={"src/tracks.sql": QUIETER})
    error = _rejects(
        "COPY (SELECT you.tracks.quieter(f.audio[1], 0.5) FROM input('film.mkv') f) TO 'out.mkv'",
        _packages(tmp_path),
        ErrorCode.UNKNOWN_FUNCTION,
        "unknown namespace 'you'",
    )
    assert error.hint == "namespaces this project can call: me"


def test_a_one_segment_qualifiers_unknown_namespace_hints_the_explicit_form(
    tmp_path: Path,
) -> None:
    """The two-segment form is never an alias lookup, so its hint is not a did-you-mean."""
    _project(tmp_path, files={"src/tracks.sql": QUIETER})
    error = _rejects(
        "COPY (SELECT you.quieter(f.audio[1], 0.5) FROM input('film.mkv') f) TO 'out.mkv'",
        _packages(tmp_path),
        ErrorCode.UNKNOWN_FUNCTION,
        "unknown namespace 'you'",
    )
    assert error.hint == "calls across packages are written <namespace>.<package>.<member>"


def test_a_near_miss_namespace_gets_a_did_you_mean(tmp_path: Path) -> None:
    """Also the three-segment form: a did-you-mean is only offered there."""
    _project(
        tmp_path, files={"src/tracks.sql": QUIETER}, manifest={"name": "mine/edits"}
    )
    error = _rejects(
        "COPY (SELECT mien.tracks.quieter(f.audio[1], 0.5) FROM input('film.mkv') f) TO 'out.mkv'",
        _packages(tmp_path),
        ErrorCode.UNKNOWN_FUNCTION,
        "unknown namespace 'mien'",
    )
    assert error.hint == "did you mean 'mine'?"


def test_an_unknown_member_gets_a_did_you_mean(tmp_path: Path) -> None:
    _project(tmp_path, files={"src/tracks.sql": QUIETER})
    error = _rejects(
        "COPY (SELECT me.edits.quiter(f.audio[1], 0.5) FROM input('film.mkv') f) TO 'out.mkv'",
        _packages(tmp_path),
        ErrorCode.UNKNOWN_FUNCTION,
        "package 'me/edits' has no export 'quiter'",
    )
    assert error.hint == "did you mean me.edits.quieter()?"


def test_an_export_naming_no_file_is_refused(tmp_path: Path) -> None:
    manifest = _project(
        tmp_path,
        files={"src/tracks.sql": QUIETER},
        manifest={"lib": {"quieter": "lib/tracks.sql"}},
    )
    with pytest.raises(FfrwdError) as caught:
        read_manifest(manifest)
    assert "export 'quieter' names no file: 'lib/tracks.sql'" in caught.value.message


def test_an_export_leaving_the_project_is_refused(tmp_path: Path) -> None:
    manifest = _project(
        tmp_path,
        files={"src/tracks.sql": QUIETER},
        manifest={"lib": {"quieter": "../tracks.sql"}},
    )
    with pytest.raises(FfrwdError) as caught:
        read_manifest(manifest)
    assert "leaves the project directory" in caught.value.message


def test_an_export_that_is_a_pattern_is_refused(tmp_path: Path) -> None:
    manifest = _project(
        tmp_path,
        files={"src/tracks.sql": QUIETER},
        manifest={"lib": {"quieter": "src/*.sql"}},
    )
    with pytest.raises(FfrwdError) as caught:
        read_manifest(manifest)
    assert "names the pattern 'src/*.sql', not a file" in caught.value.message


def test_a_file_that_does_not_define_its_export_is_refused(tmp_path: Path) -> None:
    _project(
        tmp_path,
        files={"src/tracks.sql": QUIETER},
        manifest={"lib": {"louder": "src/tracks.sql"}},
    )
    error = _rejects(
        "COPY (SELECT me.edits.louder(f.audio[1], 2) FROM input('film.mkv') f) TO 'out.mkv'",
        _packages(tmp_path),
        ErrorCode.UNSUPPORTED_SQL,
        "does not define 'louder'",
    )
    assert "tracks.sql" in error.message


def test_a_lib_file_that_does_not_define_the_default_is_refused(tmp_path: Path) -> None:
    manifest = _project(
        tmp_path,
        files={"src/tracks.sql": QUIETER},
        manifest={"lib": "src/tracks.sql"},
    )
    with pytest.raises(FfrwdError) as caught:
        package_signatures(read_manifest(manifest))
    assert "does not define 'edits'" in caught.value.message
    assert "package segment" in (caught.value.hint or "")


def test_one_name_defined_twice_across_lib_files_is_refused(tmp_path: Path) -> None:
    _project(
        tmp_path,
        files={"src/a.sql": QUIETER, "src/b.sql": QUIETER + NORMALIZE},
        manifest={"lib": {"quieter": "src/a.sql", "normalize_lang": "src/b.sql"}},
    )
    error = _rejects(
        "COPY (SELECT me.edits.quieter(f.audio[1], 0.5) FROM input('film.mkv') f) TO 'out.mkv'",
        _packages(tmp_path),
        ErrorCode.UNSUPPORTED_SQL,
        "package 'me/edits' defines 'quieter' twice",
    )
    assert "a.sql" in error.message and "b.sql" in error.message


def test_a_lib_file_that_fails_to_parse_names_the_file(tmp_path: Path) -> None:
    _project(
        tmp_path,
        files={"src/tracks.sql": "CREATE FUNCTION oops("},
        manifest={"lib": {"quieter": "src/tracks.sql"}},
    )
    error = _rejects(
        "COPY (SELECT me.edits.quieter(f.audio[1], 0.5) FROM input('film.mkv') f) TO 'out.mkv'",
        _packages(tmp_path),
        ErrorCode.PARSE_ERROR,
        "tracks.sql",
    )
    # The lib file's own line means nothing in the query, so the anchor is
    # the call that reached for it.
    assert (error.line, error.col) == (1, 14)


def test_a_lib_file_holding_a_query_is_refused(tmp_path: Path) -> None:
    _project(tmp_path, files={"src/tracks.sql": QUIETER + "SELECT 1;"})
    _rejects(
        "COPY (SELECT me.edits.quieter(f.audio[1], 0.5) FROM input('film.mkv') f) TO 'out.mkv'",
        _packages(tmp_path),
        ErrorCode.UNSUPPORTED_SQL,
        "is not a CREATE FUNCTION",
    )


def test_a_value_function_called_in_from_is_refused(tmp_path: Path) -> None:
    _project(tmp_path, files={"src/tracks.sql": NORMALIZE})
    _rejects(
        "COPY (SELECT t.x FROM me.edits.normalize_lang('en') t) TO 'out.mkv'",
        _packages(tmp_path),
        ErrorCode.UNSUPPORTED_SQL,
        "function 'me.edits.normalize_lang' returns a value, not a table",
    )


def test_a_table_function_called_as_a_value_is_refused(tmp_path: Path) -> None:
    _project(tmp_path, files={"src/tracks.sql": PICK})
    _rejects(
        "COPY (SELECT me.edits.pick('a.mka') FROM input('film.mkv') f) TO 'out.mkv'",
        _packages(tmp_path),
        ErrorCode.UNSUPPORTED_SQL,
        "function 'me.edits.pick' returns a table, not a value",
    )


def test_the_wrong_argument_count_names_the_qualified_signature(tmp_path: Path) -> None:
    _project(tmp_path, files={"src/tracks.sql": QUIETER})
    error = _rejects(
        "COPY (SELECT me.edits.quieter(f.audio[1]) FROM input('film.mkv') f) TO 'out.mkv'",
        _packages(tmp_path),
        ErrorCode.UDF_ARG_TYPE,
        "me.edits.quieter() got 1 argument, but its parameter 'factor' has no DEFAULT",
    )
    assert error.hint == "me.edits.quieter(track audio_stream, factor number) RETURNS audio_stream"


def test_two_packages_share_one_namespace_and_resolve_by_full_path(tmp_path: Path) -> None:
    """The real rule: `ns.pkg.member` reaches the right package even when `ns` holds several."""
    first = _library(tmp_path / "one", "me", "0.5", package="alpha")
    second = _library(tmp_path / "two", "me", "0.25", package="beta")
    project = tmp_path / "work"
    _project(project, files={}, manifest={"name": "other/edits"})
    _lock(project, [_link(first), _link(second)])
    packages = _packages(project)
    alpha = _argv(
        "COPY (SELECT me.alpha.quieter(f.audio[1]) FROM input('film.mkv') f) TO 'out.mkv'",
        packages,
    )
    beta = _argv(
        "COPY (SELECT me.beta.quieter(f.audio[1]) FROM input('film.mkv') f) TO 'out.mkv'",
        packages,
    )
    assert "volume=volume=0.5" in " ".join(alpha)
    assert "volume=volume=0.25" in " ".join(beta)


def test_a_two_segment_call_naming_no_package_in_a_shared_namespace_says_what_it_holds(
    tmp_path: Path,
) -> None:
    """`me.quieter` names no `me/quieter` package; `me` holds two others."""
    first = _library(tmp_path / "one", "me", "0.5", package="alpha")
    second = _library(tmp_path / "two", "me", "0.25", package="beta")
    project = tmp_path / "work"
    _project(project, files={}, manifest={"name": "other/edits"})
    _lock(project, [_link(first), _link(second)])
    error = _rejects(
        "COPY (SELECT me.quieter(f.audio[1]) FROM input('film.mkv') f) TO 'out.mkv'",
        _packages(project),
        ErrorCode.UNKNOWN_FUNCTION,
        "namespace 'me' has no package 'quieter'",
    )
    assert error.hint == "me holds: alpha, beta"


# ---------------------------------------------------------------------------
# manifest validation
# ---------------------------------------------------------------------------


def test_a_manifest_that_is_not_json_is_anchored(tmp_path: Path) -> None:
    manifest = _project(tmp_path, files={"src/tracks.sql": QUIETER}, text='{\n  "name",\n}\n')
    with pytest.raises(FfrwdError) as caught:
        read_manifest(manifest)
    assert "is not valid JSON" in caught.value.message
    assert caught.value.line == 2


def test_a_manifest_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    manifest = _project(tmp_path, files={"src/tracks.sql": QUIETER}, text="[]\n")
    with pytest.raises(FfrwdError) as caught:
        read_manifest(manifest)
    assert "is not a JSON object" in caught.value.message


@pytest.mark.parametrize("missing", ["name", "version"])
def test_every_required_key_is_required(tmp_path: Path, missing: str) -> None:
    declared: dict[str, object] = {"name": "me/edits", "version": "0.1.0"}
    del declared[missing]
    manifest = _project(tmp_path, text=json.dumps(declared, indent=2))
    with pytest.raises(FfrwdError) as caught:
        read_manifest(manifest)
    assert f'is missing "{missing}"' in caught.value.message


def test_a_name_must_be_two_plain_identifiers(tmp_path: Path) -> None:
    for claimed in ("me", "My/edits", "me/1st", "a-b/c", "a/b/c", "a.b/c", "", "me/"):
        manifest = _project(tmp_path, manifest={"name": claimed})
        with pytest.raises(FfrwdError) as caught:
            read_manifest(manifest)
        assert "name" in caught.value.message, claimed


def test_namespace_is_no_longer_a_key(tmp_path: Path) -> None:
    manifest = _project(tmp_path, manifest={"namespace": "me"})
    with pytest.raises(FfrwdError) as caught:
        read_manifest(manifest)
    assert "unknown key 'namespace'" in caught.value.message
    assert "the name carries the namespace" in (caught.value.hint or "")


def test_exports_is_no_longer_a_key(tmp_path: Path) -> None:
    manifest = _project(tmp_path, manifest={"exports": ["src/*.sql"]})
    with pytest.raises(FfrwdError) as caught:
        read_manifest(manifest)
    assert "unknown key 'exports'" in caught.value.message
    assert '"lib"' in (caught.value.hint or "")


@pytest.mark.parametrize("retired", ["libs", "bins"])
def test_libs_and_bins_are_no_longer_keys(tmp_path: Path, retired: str) -> None:
    """`lib`/`bin` replaced them; the hint names the singular that did."""
    singular = "lib" if retired == "libs" else "bin"
    manifest = _project(tmp_path, manifest={retired: {}})
    with pytest.raises(FfrwdError) as caught:
        read_manifest(manifest)
    assert f"unknown key {retired!r}" in caught.value.message
    assert f'"{singular}"' in (caught.value.hint or "")


def test_an_unknown_key_gets_a_did_you_mean(tmp_path: Path) -> None:
    manifest = _project(tmp_path, manifest={"binz": {}})
    with pytest.raises(FfrwdError) as caught:
        read_manifest(manifest)
    assert "unknown key 'binz'" in caught.value.message
    assert caught.value.hint == "did you mean 'bin'?"


def test_a_description_and_dependencies_are_accepted(tmp_path: Path) -> None:
    manifest = _project(
        tmp_path,
        files={"src/tracks.sql": QUIETER},
        manifest={
            "description": "edits",
            "dependencies": {"broadcast/tracks": "^1.2.0"},
        },
    )
    package = read_manifest(manifest)
    assert package.name == "me/edits"
    assert package.namespace == "me" and package.package == "edits"
    assert package.version == "0.1.0"
    assert [path.name for path in package.exports.values()] == ["tracks.sql"]
    assert package.dependencies == {"broadcast/tracks": "^1.2.0"}


def test_a_map_libs_members_are_named_and_ordered(tmp_path: Path) -> None:
    """A map `lib` has no default: every member is named, in the manifest's own order."""
    brighten = QUIETER.replace("quieter", "brighten")
    manifest = _project(
        tmp_path,
        files={"src/other.sql": brighten, "src/tracks.sql": QUIETER},
        manifest={"lib": {"brighten": "src/other.sql", "quieter": "src/tracks.sql"}},
    )
    package = read_manifest(manifest)
    assert list(package.exports) == ["brighten", "quieter"]
    assert package.export("brighten") == tmp_path / "src" / "other.sql"
    assert package.export("quieter") == tmp_path / "src" / "tracks.sql"
    assert package.export("nothing") is None
    assert package.export() is None  # a map has no root-callable member


def test_a_map_member_may_carry_the_packages_own_name(tmp_path: Path) -> None:
    """The growth path: a one-recipe package adds a second and keeps the name.

    The member is reached at three segments like any other; only the string
    form answers at the package's own name.
    """
    manifest = _project(
        tmp_path,
        files={"queries/split.sql": RECIPE, "queries/other.sql": RECIPE},
        manifest={"bin": {"edits": "queries/split.sql", "other": "queries/other.sql"}},
    )
    package = read_manifest(manifest)
    assert package.recipe("edits") == manifest.parent / "queries" / "split.sql"
    assert package.recipe() is None


def test_a_string_bin_answers_at_the_packages_own_name(tmp_path: Path) -> None:
    manifest = _project(
        tmp_path,
        files={"queries/split.sql": RECIPE},
        manifest={"bin": "queries/split.sql"},
    )
    package = read_manifest(manifest)
    assert package.recipe() == package.recipe("edits")


@pytest.mark.parametrize("key", ["Tracks", "a-b", "tracks", ""])
def test_a_dependency_key_must_be_a_package_name(tmp_path: Path, key: str) -> None:
    manifest = _project(tmp_path, manifest={"dependencies": {key: "^1.0.0"}})
    with pytest.raises(FfrwdError) as caught:
        read_manifest(manifest)
    assert f"dependency key {key!r} is not a package name" in caught.value.message


def test_a_dependency_under_a_reserved_namespace_is_refused(tmp_path: Path) -> None:
    manifest = _project(tmp_path, manifest={"dependencies": {"ffmpeg/tracks": "^1.0.0"}})
    with pytest.raises(FfrwdError) as caught:
        read_manifest(manifest)
    assert "namespace 'ffmpeg' is reserved" in caught.value.message


def test_a_dependency_value_must_be_a_non_empty_string(tmp_path: Path) -> None:
    for written in ("", 7, None):
        manifest = _project(tmp_path, manifest={"dependencies": {"broadcast/tracks": written}})
        with pytest.raises(FfrwdError) as caught:
            read_manifest(manifest)
        assert "dependency 'broadcast/tracks' must be a string" in caught.value.message, written


# ---------------------------------------------------------------------------
# what a package provides: exports, recipes, or neither
# ---------------------------------------------------------------------------

RECIPE = (
    "-- variables: source (input media path), dest (output path)\n"
    "COPY (SELECT f.video[1] FROM input(:'source') f) TO :'dest';\n"
)

_BIN = {"bin": {"split-chapters": "queries/split.sql"}}


def _manifest_text(**declared: object) -> str:
    """A manifest written key by key, so a rejection's line is predictable."""
    return json.dumps({"name": "me/edits", "version": "0.1.0", **declared}, indent=2)


def test_a_manifest_declaring_neither_half_is_a_package(tmp_path: Path) -> None:
    """The consumer project: a name and its dependencies, nothing provided."""
    manifest = _project(
        tmp_path, text=_manifest_text(dependencies={"broadcast/tracks": "^1.2.0"})
    )
    package = read_manifest(manifest)
    assert dict(package.exports) == {}
    assert dict(package.recipes) == {}


def test_a_map_bin_declares_named_recipes_beside_the_exports(tmp_path: Path) -> None:
    manifest = _project(
        tmp_path,
        files={"src/tracks.sql": QUIETER, "queries/split.sql": RECIPE},
        manifest=_BIN,
    )
    package = read_manifest(manifest)
    assert list(package.recipes) == ["split-chapters"]
    assert package.recipe("split-chapters") == tmp_path / "queries" / "split.sql"
    assert package.recipe("nothing-like-it") is None


def test_bin_declares_the_default_recipe(tmp_path: Path) -> None:
    manifest = _project(
        tmp_path,
        files={"queries/split.sql": RECIPE},
        text=_manifest_text(bin="queries/split.sql"),
    )
    package = read_manifest(manifest)
    assert list(package.recipes) == ["edits"]
    assert package.recipe() == tmp_path / "queries" / "split.sql"


def test_a_package_may_ship_recipes_and_export_nothing(tmp_path: Path) -> None:
    manifest = _project(
        tmp_path, files={"queries/split.sql": RECIPE}, text=_manifest_text(**_BIN)
    )
    package = read_manifest(manifest)
    assert dict(package.exports) == {}
    assert list(package.recipes) == ["split-chapters"]


def test_a_recipe_is_a_query_and_the_lib_rule_never_reaches_it(tmp_path: Path) -> None:
    """A bin file holds a whole query -- the rule that rejects one in a lib
    file is about lib files, and a compile that resolves into the package proves it."""
    _project(
        tmp_path,
        files={"src/tracks.sql": QUIETER, "queries/split.sql": RECIPE},
        manifest=_BIN,
    )
    argv = _argv(
        "COPY (SELECT me.edits.quieter(f.audio[1], 0.5) FROM input('film.mkv') f) TO 'out.mkv'",
        _packages(tmp_path),
    )
    assert "volume=volume=0.5" in " ".join(argv)


@pytest.mark.parametrize("claimed", ["Split", "1st", "-split", "split chapters", "", "split.sh"])
def test_a_recipe_name_is_a_command_name(tmp_path: Path, claimed: str) -> None:
    manifest = _project(
        tmp_path,
        files={"queries/split.sql": RECIPE},
        text=_manifest_text(bin={claimed: "queries/split.sql"}),
    )
    with pytest.raises(FfrwdError) as caught:
        read_manifest(manifest)
    assert f"recipe name {claimed!r} is not a command name" in caught.value.message
    assert caught.value.line == 5


def test_bin_that_is_neither_a_string_nor_an_object_is_refused(tmp_path: Path) -> None:
    manifest = _project(
        tmp_path,
        files={"queries/split.sql": RECIPE},
        text=_manifest_text(bin=["queries/split.sql"]),
    )
    with pytest.raises(FfrwdError) as caught:
        read_manifest(manifest)
    assert '"bin" must be a string or a JSON object' in caught.value.message


def test_bin_may_be_a_map_with_several_named_recipes(tmp_path: Path) -> None:
    """The old shape rejected a map under `bin`; it is now the map form itself."""
    manifest = _project(
        tmp_path,
        files={"queries/split.sql": RECIPE},
        text=_manifest_text(bin={"split": "queries/split.sql"}),
    )
    package = read_manifest(manifest)
    assert list(package.recipes) == ["split"]
    assert package.recipe("split") == tmp_path / "queries" / "split.sql"
    assert package.recipe() is None  # a map has no root-callable member


def test_a_recipe_that_names_no_string_is_refused(tmp_path: Path) -> None:
    manifest = _project(
        tmp_path, files={"queries/split.sql": RECIPE}, text=_manifest_text(bin={"split": 1})
    )
    with pytest.raises(FfrwdError) as caught:
        read_manifest(manifest)
    assert "recipe 'split' must name one file" in caught.value.message


def test_a_recipe_leaving_the_project_is_refused(tmp_path: Path) -> None:
    manifest = _project(
        tmp_path,
        files={"queries/split.sql": RECIPE},
        text=_manifest_text(bin={"split": "../split.sql"}),
    )
    with pytest.raises(FfrwdError) as caught:
        read_manifest(manifest)
    assert "leaves the project directory" in caught.value.message


def test_a_recipe_matching_no_file_is_refused(tmp_path: Path) -> None:
    manifest = _project(
        tmp_path,
        files={"queries/split.sql": RECIPE},
        text=_manifest_text(bin={"split": "queries/gone.sql"}),
    )
    with pytest.raises(FfrwdError) as caught:
        read_manifest(manifest)
    assert "recipe 'split' names no file: 'queries/gone.sql'" in caught.value.message


def test_a_recipe_that_is_a_pattern_is_refused(tmp_path: Path) -> None:
    manifest = _project(
        tmp_path,
        files={"queries/split.sql": RECIPE},
        text=_manifest_text(bin={"split": "queries/*.sql"}),
    )
    with pytest.raises(FfrwdError) as caught:
        read_manifest(manifest)
    assert "names the pattern 'queries/*.sql', not a file" in caught.value.message


def test_two_recipes_written_under_one_name_are_refused(tmp_path: Path) -> None:
    """``json`` keeps the last of two same-named keys, which would drop one silently."""
    manifest = _project(
        tmp_path,
        files={"queries/split.sql": RECIPE, "queries/other.sql": RECIPE},
        text=(
            '{\n  "name": "me/edits",\n  "version": "0.1.0",\n'
            '  "bin": {\n    "split": "queries/split.sql",\n'
            '    "split": "queries/other.sql"\n  }\n}\n'
        ),
    )
    with pytest.raises(FfrwdError) as caught:
        read_manifest(manifest)
    assert "bin declares 'split' twice" in caught.value.message
    assert caught.value.line == 5


def test_the_signatures_a_package_exports_are_readable_without_a_query(tmp_path: Path) -> None:
    manifest = _project(tmp_path, files={"src/tracks.sql": QUIETER + NORMALIZE})
    signatures = package_signatures(read_manifest(manifest))
    assert [signature.written for signature in signatures] == [
        "quieter(track audio_stream, factor number)",
        "normalize_lang(raw text)",
    ]
    assert [signature.package for signature in signatures] == ["me/edits", "me/edits"]
    assert [signature.returns for signature in signatures] == ["audio_stream", "text"]
    assert [signature.export.name for signature in signatures] == ["tracks.sql", "tracks.sql"]


def test_reading_the_signatures_of_a_broken_lib_file_is_refused(tmp_path: Path) -> None:
    manifest = _project(tmp_path, files={"src/tracks.sql": QUIETER + "SELECT 1;"})
    with pytest.raises(FfrwdError) as caught:
        package_signatures(read_manifest(manifest))
    assert "is not a CREATE FUNCTION" in caught.value.message
    assert "src" in caught.value.message


# ---------------------------------------------------------------------------
# a qualifier owns the name under it
# ---------------------------------------------------------------------------


def test_a_script_function_does_not_shadow_a_filter_call() -> None:
    """`ffmpeg.<name>` is the installed ffmpeg's, whatever the script defines.

    The definition stays uncalled, which is what says the qualified call never
    reached it -- inlining it would have marked it used and compiled.
    """
    definition = (
        "CREATE FUNCTION quiet(x audio_stream) RETURNS audio_stream AS $$\n"
        "  SELECT volume(x, 0.5)\n"
        "$$ LANGUAGE sql;\n"
    )
    _rejects(
        definition + "COPY (SELECT ffmpeg.quiet(f.audio[1]) FROM input('a.mp4') f) TO 'o.mp4'",
        None,
        ErrorCode.UNSUPPORTED_SQL,
        "function 'quiet' is never called",
    )


def test_a_script_function_does_not_shadow_a_generated_source() -> None:
    definition = (
        "CREATE FUNCTION testsrc(path text) RETURNS TABLE(v video_stream) AS $$\n"
        "  SELECT f.video[1] FROM input(path) f\n"
        "$$ LANGUAGE sql;\n"
    )
    _rejects(
        definition + "COPY (SELECT t.video[1] FROM ffmpeg.testsrc(duration => 2) t) TO 'o.mp4'",
        None,
        ErrorCode.UNSUPPORTED_SQL,
        "'testsrc' is never called",
    )


# ---------------------------------------------------------------------------
# the CLI derives the project from -f's path, or the working directory
# ---------------------------------------------------------------------------


def test_the_cli_finds_the_project_above_the_query_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _project(tmp_path, files={"src/tracks.sql": QUIETER})
    query = tmp_path / "queries" / "out.sql"
    query.parent.mkdir()
    query.write_text(
        "COPY (SELECT me.edits.quieter(f.audio[1], 0.5) FROM input('film.mkv') f) TO 'out.mkv'",
        encoding="utf-8",
    )
    assert cli.main(["compile", "-f", str(query)]) == 0
    assert "volume=volume=0.5" in capsys.readouterr().out


def test_the_cli_finds_the_project_above_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _project(tmp_path, files={"src/tracks.sql": QUIETER})
    deep = tmp_path / "queries"
    deep.mkdir()
    monkeypatch.chdir(deep)
    code = cli.main(
        [
            "compile",
            "COPY (SELECT me.edits.quieter(f.audio[1], 0.5) FROM input('f.mkv') f) TO 'o.mkv'",
        ]
    )
    assert code == 0
    assert "volume=volume=0.5" in capsys.readouterr().out


def test_the_cli_reports_a_malformed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _project(tmp_path, files={"src/tracks.sql": QUIETER}, text="{ nope\n")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["validate", "SELECT f.video[1] FROM input('f.mkv') f"]) == 1
    assert "ffrwd.json" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# the MCP tools take the project as an argument, never from the process
# ---------------------------------------------------------------------------


def test_the_mcp_tools_resolve_against_the_named_project(tmp_path: Path) -> None:
    _project(tmp_path, files={"src/tracks.sql": QUIETER})
    query = "COPY (SELECT me.edits.quieter(f.audio[1], 0.5) FROM input('f.mkv') f) TO 'o.mkv'"
    assert mcp_tools.validate_query(query, None, str(tmp_path)) == {}
    result = mcp_tools.compile_query(query, None, str(tmp_path))
    assert "volume=volume=0.5" in result["filter_complex"][0]


def test_the_mcp_tools_see_no_project_without_one(tmp_path: Path) -> None:
    _project(tmp_path, files={"src/tracks.sql": QUIETER})
    query = "COPY (SELECT me.edits.quieter(f.audio[1], 0.5) FROM input('f.mkv') f) TO 'o.mkv'"
    assert mcp_tools.validate_query(query)["code"] == ErrorCode.UNSUPPORTED_SQL.value


def test_a_malformed_manifest_is_data_for_validate(tmp_path: Path) -> None:
    _project(tmp_path, files={"src/tracks.sql": QUIETER}, text="{ nope\n")
    error = mcp_tools.validate_query(
        "SELECT f.video[1] FROM input('f.mkv') f", None, str(tmp_path)
    )
    assert "ffrwd.json" in error["message"]


# ---------------------------------------------------------------------------
# the lockfile layers: the store, links, and which layer answers
# ---------------------------------------------------------------------------

QUERY = "COPY (SELECT {call}(f.audio[1]) FROM input('film.mkv') f) TO 'out.mkv'"


def _quieter(factor: str) -> str:
    """A one-argument ``quieter`` whose factor shows up in the filter graph.

    Which layer answered is then readable straight off the compiled command.
    """
    return (
        "CREATE FUNCTION quieter(track audio_stream) RETURNS audio_stream AS $$\n"
        f"  SELECT volume(track, {factor})\n"
        "$$ LANGUAGE sql;\n"
    )


def _library(
    root: Path,
    namespace: str,
    factor: str,
    *,
    version: str = "1.0.0",
    package: str = "lib",
    member: str = "quieter",
    src: str | None = None,
    dependencies: dict[str, str] | None = None,
) -> Path:
    """A package directory of its own: a manifest named `namespace`/`package`, one source.

    `src` overrides the default `quieter(factor)` body and `member` its
    exported name -- for a package whose own body calls into ANOTHER package.
    `dependencies` is its manifest's own.
    """
    (root / "src").mkdir(parents=True, exist_ok=True)
    body = src if src is not None else _quieter(factor)
    (root / "src" / "lib.sql").write_text(body, encoding="utf-8")
    declared: dict[str, object] = {
        "name": f"{namespace}/{package}",
        "version": version,
        "lib": {member: "src/lib.sql"},
    }
    if dependencies:
        declared["dependencies"] = dependencies
    (root / "ffrwd.json").write_text(json.dumps(declared) + "\n", encoding="utf-8")
    return root


def _digest(archive: bytes) -> str:
    return hashlib.sha256(archive).hexdigest()


def _installed(source: Path, *, dependencies: dict[str, str] | None = None) -> dict[str, object]:
    """Put a package in the store the way installing does; return the entry that pins it.

    `dependencies` overrides what the entry itself records as ITS OWN
    resolved dependencies -- the same shape `install` would write walking the
    package's manifest, for a test that does not go through `install` itself.
    """
    package = read_manifest(source / "ffrwd.json")
    archive = store.pack(source)
    sha256 = _digest(archive)
    store.unpack(package.name, archive, sha256)
    entry: dict[str, object] = {
        "kind": "registry",
        "name": package.name,
        "version": package.version,
        "sha256": sha256,
        "store": store.entry_path(sha256),
    }
    if dependencies:
        entry["dependencies"] = dependencies
    return entry


def _link(directory: Path) -> dict[str, object]:
    return {"kind": "link", "path": str(directory)}


def _lock(
    directory: Path,
    entries: list[dict[str, object]],
    *,
    reproducible: bool | None = None,
    text: str | None = None,
    dependencies: dict[str, str] | None = None,
) -> Path:
    """Write a lockfile the way installing would, and return its path.

    `dependencies` is what the lockfile's own top-level field would hold: the
    project's own directly-installed package name to exact version.
    """
    path = directory / "ffrwd.lock"
    directory.mkdir(parents=True, exist_ok=True)
    if text is not None:
        path.write_text(text, encoding="utf-8")
        return path
    linked = [entry for entry in entries if entry.get("kind") == "link"]
    honest = not linked if reproducible is None else reproducible
    data: dict[str, object] = {"format_version": LOCK_FORMAT_VERSION, "reproducible": honest}
    if not honest:
        data["not_reproducible_because"] = (
            "a package is linked to a working directory, so its files are not pinned here"
        )
    if dependencies:
        data["dependencies"] = dependencies
    data["packages"] = entries
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def store_home(_isolated_store: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the store and the machine-wide lockfile at this test's own directory."""
    home = tmp_path / "cache"
    home.mkdir()
    monkeypatch.setattr(store, "_cache_dir", lambda: home)
    return home


def _heard(sql: str, packages: PackageSet | None) -> tuple[list[str], list[FfrwdWarning]]:
    """The compiled argv, and everything the compile had to say about it."""
    said: list[FfrwdWarning] = []
    argv = build_ffmpeg_args(emit(compile_sql(sql, packages=packages, on_warning=said.append)))
    return argv, said


def _codes(said: list[FfrwdWarning]) -> list[WarningCode]:
    return [warning.code for warning in said]


def _refuses(root: Path, needle: str) -> FfrwdError:
    with pytest.raises(FfrwdError) as caught:
        discover(root)
    assert needle in caught.value.message, caught.value.message
    return caught.value


def test_a_locked_package_resolves_out_of_the_store(store_home: Path, tmp_path: Path) -> None:
    entry = _installed(_library(tmp_path / "built", "tracks", "0.5"))
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [entry])
    argv, said = _heard(QUERY.format(call="tracks.lib.quieter"), _packages(project))
    assert argv == _argv(_quieter("0.5") + QUERY.format(call="quieter"))
    assert said == []


def test_content_missing_from_the_store_is_refused(store_home: Path, tmp_path: Path) -> None:
    entry = _installed(_library(tmp_path / "built", "tracks", "0.5"))
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [entry])
    shutil.rmtree(store.store_dir() / str(entry["store"]))
    error = _refuses(project, "is not in the store")
    assert "tracks/lib" in error.message


def test_a_store_path_from_another_layout_is_refused(store_home: Path, tmp_path: Path) -> None:
    entry = _installed(_library(tmp_path / "built", "tracks", "0.5"))
    entry["store"] = str(entry["store"]).replace(store.STORE_FORMAT, "v99", 1)
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [entry])
    error = _refuses(project, "store format")
    assert "v99" in error.message


def test_a_store_path_that_is_not_the_digest_s_own_is_refused(
    store_home: Path, tmp_path: Path
) -> None:
    """An entry pinning one digest while pointing at another's content."""
    entry = _installed(_library(tmp_path / "built", "tracks", "0.5"))
    other = _installed(_library(tmp_path / "other", "far", "0.1"))
    entry["store"] = other["store"]
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [entry])
    error = _refuses(project, "is not where content of")
    assert str(entry["sha256"]) in error.message


def test_a_lockfile_entry_the_stored_package_disagrees_with_is_refused(
    store_home: Path, tmp_path: Path
) -> None:
    entry = _installed(_library(tmp_path / "built", "tracks", "0.5"))
    entry["version"] = "9.9.9"
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [entry])
    _refuses(project, "records version '9.9.9'")


# ---------------------------------------------------------------------------
# the archive: packing, verifying, extracting
# ---------------------------------------------------------------------------


def _archive(add: Callable[[tarfile.TarFile], None]) -> tuple[bytes, str]:
    """A gzipped tar built member by member, and the digest of its bytes."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as archive:
        add(archive)
    data = raw.getvalue()
    return data, _digest(data)


def _one_member(
    name: str, *, kind: bytes = tarfile.REGTYPE, linkname: str = ""
) -> tuple[bytes, str]:
    """An archive holding one member of the given name and kind."""

    def add(archive: tarfile.TarFile) -> None:
        info = tarfile.TarInfo(name)
        info.type = kind
        info.linkname = linkname
        if kind != tarfile.REGTYPE:
            archive.addfile(info)
            return
        info.size = len(b"payload")
        archive.addfile(info, io.BytesIO(b"payload"))

    return _archive(add)


def _unpacking(archive: bytes, sha256: str) -> FfrwdError:
    with pytest.raises(FfrwdError) as caught:
        store.unpack("broadcast/tracks", archive, sha256)
    return caught.value


def _nothing_stored(sha256: str) -> None:
    """Neither the entry nor a half-written one beside it survived the rejection."""
    entry = store.store_dir() / store.entry_path(sha256)
    assert not entry.exists()
    assert not entry.parent.exists() or list(entry.parent.iterdir()) == []


_MODULE_LIB = (
    "CREATE FUNCTION depth(v video_stream) RETURNS video_stream\n"
    "  AS 'target/wasm32-wasip2/release/depth.wasm', 'depth' LANGUAGE wasm;\n"
)


def _wasm_package(root: Path, *, ignores: dict[str, str] | None = None) -> Path:
    """A wasm package whose module is build output, with junk around it.

    The manifest names the wasm under ``target/``; every ignore file the test
    wants is written at the root. ``.git/`` and ``build.log`` and ``notes/``
    are here to be left out.
    """
    for relative, text in {
        "src/lib.sql": _MODULE_LIB,
        "recipes/run.sql": "COPY (SELECT f.video[1] FROM input('in.mkv') f) TO 'out.mkv';\n",
        "README.md": "# depth\n\nA <em>depth</em> module.\n",
        "notes/design.md": "scratch\n",
        "build.log": "warning: nothing\n",
        ".git/HEAD": "ref: refs/heads/main\n",
        "target/debug/junk.txt": "debug build\n",
        "target/wasm32-wasip2/release/depth.d": "depth.wasm: src/lib.rs\n",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    module = root / "target" / "wasm32-wasip2" / "release" / "depth.wasm"
    module.write_bytes(b"\0asm\x01\x00\x00\x00")
    for name, text in (ignores or {}).items():
        (root / name).write_text(text, encoding="utf-8")
    (root / "ffrwd.json").write_text(
        json.dumps(
            {
                "name": "broadcast/depth",
                "version": "1.0.0",
                "lib": {"depth": "src/lib.sql"},
                "bin": {"run": "recipes/run.sql"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _members(archive: bytes) -> list[str]:
    """Every member name the archive holds, in the order it holds them."""
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as opened:
        return opened.getnames()


def test_the_closure_ships_out_of_an_ignored_build_directory(tmp_path: Path) -> None:
    """The whole member list: what the manifest names, and nothing else excluded.

    ``target/`` is excluded, so the archive carries exactly the one wasm the
    lib declares and none of its siblings; ``notes/`` and ``build.log`` go by
    the ignore file; ``.git/`` and ``.ffrwdignore`` itself go for being
    dot-entries. The directories above the pulled-back module ship with it.
    """
    source = _wasm_package(
        tmp_path / "built", ignores={".ffrwdignore": "# junk\n\nnotes/\n*.log\ntarget/\n"}
    )
    assert _members(store.pack(source)) == [
        "README.md",
        "ffrwd.json",
        "recipes",
        "recipes/run.sql",
        "src",
        "src/lib.sql",
        "target",
        "target/wasm32-wasip2",
        "target/wasm32-wasip2/release",
        "target/wasm32-wasip2/release/depth.wasm",
    ]


def test_a_stock_rust_gitignore_excludes_the_build_and_the_closure_still_ships(
    tmp_path: Path,
) -> None:
    """What `cargo new` writes is enough: no .ffrwdignore, and the wasm travels."""
    source = _wasm_package(tmp_path / "built", ignores={".gitignore": "/target\n"})
    members = _members(store.pack(source))
    assert "target/wasm32-wasip2/release/depth.wasm" in members
    assert "target/debug/junk.txt" not in members
    assert "target/wasm32-wasip2/release/depth.d" not in members


def test_the_two_ignore_files_union_rather_than_replace(tmp_path: Path) -> None:
    """Each file alone excludes its own; together they exclude both sets."""
    ours = {".ffrwdignore": "notes/\n"}
    theirs = {".gitignore": "*.log\n"}
    only_ours = _members(store.pack(_wasm_package(tmp_path / "ours", ignores=ours)))
    only_theirs = _members(store.pack(_wasm_package(tmp_path / "theirs", ignores=theirs)))
    both = _members(store.pack(_wasm_package(tmp_path / "both", ignores={**ours, **theirs})))

    assert "notes/design.md" not in only_ours and "build.log" in only_ours
    assert "build.log" not in only_theirs and "notes/design.md" in only_theirs
    assert "notes/design.md" not in both and "build.log" not in both


def test_a_package_with_no_ignore_file_ships_everything_but_the_dot_entries(
    tmp_path: Path,
) -> None:
    source = _wasm_package(tmp_path / "built")
    members = _members(store.pack(source))
    assert "notes/design.md" in members
    assert "build.log" in members
    assert "target/debug/junk.txt" in members
    assert not [name for name in members if name.split("/")[0].startswith(".")]


def test_a_negation_in_our_ignore_file_is_refused_naming_the_line(tmp_path: Path) -> None:
    source = _wasm_package(
        tmp_path / "built", ignores={".ffrwdignore": "target/\n!target/keep.wasm\n"}
    )
    with pytest.raises(FfrwdError) as caught:
        store.pack(source)
    assert "line 2: '!target/keep.wasm' negates a pattern" in caught.value.message
    assert "nothing to negate" in (caught.value.hint or "")


def test_a_negation_in_a_borrowed_gitignore_is_skipped_and_warned_about(
    tmp_path: Path,
) -> None:
    """A user's gitignore is not ours to refuse: the line is dropped, the pack runs."""
    source = _wasm_package(
        tmp_path / "built", ignores={".gitignore": "*.log\n!keep.log\n"}
    )
    said: list[FfrwdWarning] = []
    members = _members(store.pack(source, on_warning=said.append))
    assert "build.log" not in members
    assert _codes(said) == [WarningCode.IGNORE_PATTERN]
    assert said[0].message == ".gitignore line 2: '!keep.log' negates a pattern, and was skipped"
    assert said[0].line == 2


def test_packing_a_tree_twice_produces_the_same_bytes(tmp_path: Path) -> None:
    """Determinism over the tree the ignore rules and the closure both act on."""
    source = _wasm_package(
        tmp_path / "built", ignores={".ffrwdignore": "notes/\n", ".gitignore": "/target\n"}
    )
    assert store.pack(source) == store.pack(source)


def test_a_directory_with_no_manifest_packs_with_no_closure(tmp_path: Path) -> None:
    """No manifest, so nothing is essential -- and the dot-entries still go."""
    source = tmp_path / "plain"
    (source / "src").mkdir(parents=True)
    (source / "src" / "notes.txt").write_text("kept\n", encoding="utf-8")
    (source / ".hidden").write_text("dropped\n", encoding="utf-8")
    (source / ".ffrwdignore").write_text("*.tmp\n", encoding="utf-8")
    (source / "scratch.tmp").write_text("dropped\n", encoding="utf-8")
    assert _members(store.pack(source)) == ["src", "src/notes.txt"]


def test_packing_the_same_content_from_two_directories_produces_the_same_bytes(
    tmp_path: Path,
) -> None:
    first = _library(tmp_path / "one", "tracks", "0.5")
    second = _library(tmp_path / "two", "tracks", "0.5")
    assert store.pack(first) == store.pack(second)


def test_packing_a_tree_holding_a_link_is_refused(tmp_path: Path) -> None:
    source = _library(tmp_path / "built", "tracks", "0.5")
    try:
        (source / "src" / "elsewhere.sql").symlink_to(tmp_path / "outside.sql")
    except (NotImplementedError, OSError) as err:  # a platform that will not make one
        pytest.skip(f"symlinks unavailable: {err}")
    with pytest.raises(FfrwdError) as caught:
        store.pack(source)
    assert "regular files and directories only" in caught.value.message


def test_a_verified_archive_unpacks_into_the_store(store_home: Path, tmp_path: Path) -> None:
    source = _library(tmp_path / "built", "tracks", "0.5")
    archive = store.pack(source)
    sha256 = _digest(archive)
    stored = store.unpack("tracks/lib", archive, sha256)
    assert stored == store.store_dir() / store.entry_path(sha256)
    assert (stored / "src" / "lib.sql").read_text(encoding="utf-8") == _quieter("0.5")
    read_manifest(stored / "ffrwd.json")


def test_a_wasm_module_round_trips_through_the_archive_byte_for_byte(
    store_home: Path, tmp_path: Path
) -> None:
    """A package's modules ship with it, so the archive carries them unchanged."""
    source = _library(tmp_path / "built", "tracks", "0.5")
    module = bytes(range(256)) * 64  # every byte value, nothing text-shaped
    (source / "modules").mkdir()
    (source / "modules" / "invert.wasm").write_bytes(module)
    archive = store.pack(source)
    stored = store.unpack("tracks/lib", archive, _digest(archive))
    assert (stored / "modules" / "invert.wasm").read_bytes() == module


def test_an_archive_that_is_not_what_was_pinned_is_refused(
    store_home: Path, tmp_path: Path
) -> None:
    pinned = _digest(store.pack(_library(tmp_path / "built", "tracks", "0.5")))
    swapped = store.pack(_library(tmp_path / "other", "tracks", "0.25"))
    error = _unpacking(swapped, pinned)
    assert "hashes to" in error.message
    assert pinned in error.message
    _nothing_stored(pinned)


def test_bytes_that_are_not_an_archive_are_refused_by_their_digest(store_home: Path) -> None:
    # The rejection names the digest, not a malformed tar: the bytes were never
    # handed to an unpacker.
    pinned = "b" * 64
    error = _unpacking(b"not an archive at all", pinned)
    assert "hashes to" in error.message
    _nothing_stored(pinned)


def test_unpacking_a_digest_already_in_the_store_leaves_it_alone(
    store_home: Path, tmp_path: Path
) -> None:
    archive = store.pack(_library(tmp_path / "built", "tracks", "0.5"))
    sha256 = _digest(archive)
    stored = store.unpack("tracks/lib", archive, sha256)
    (stored / "marker").write_text("kept", encoding="utf-8")
    assert store.unpack("tracks/lib", archive, sha256) == stored
    assert (stored / "marker").read_text(encoding="utf-8") == "kept"


def test_a_member_that_leaves_the_destination_is_refused(store_home: Path) -> None:
    for name in (
        '../escape.sql',
        '/etc/escape.sql',
        '..',
        'src/../../escape.sql',
        '..\\escape.sql',
    ):
        archive, sha256 = _one_member(name)
        error = _unpacking(archive, sha256)
        assert "leaves the directory" in error.message, name
        _nothing_stored(sha256)


def test_a_member_that_is_not_a_file_or_a_directory_is_refused(store_home: Path) -> None:
    for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.FIFOTYPE):
        archive, sha256 = _one_member("passwd", kind=kind, linkname="src/lib.sql")
        error = _unpacking(archive, sha256)
        assert "neither a regular file nor a directory" in error.message, kind
        _nothing_stored(sha256)


def test_an_archive_unpacking_past_the_size_cap_is_refused(
    store_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store, "_MAX_UNPACKED_BYTES", 64)

    def add(archive: tarfile.TarFile) -> None:
        info = tarfile.TarInfo("big.sql")
        info.size = 4096
        archive.addfile(info, io.BytesIO(b"\0" * 4096))

    archive, sha256 = _archive(add)
    error = _unpacking(archive, sha256)
    assert "more than 64 bytes" in error.message
    _nothing_stored(sha256)


def test_an_archive_past_the_member_cap_is_refused(
    store_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store, "_MAX_MEMBERS", 2)

    def add(archive: tarfile.TarFile) -> None:
        for index in range(3):
            info = tarfile.TarInfo(f"{index}.sql")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))

    archive, sha256 = _archive(add)
    error = _unpacking(archive, sha256)
    assert "more than 2 members" in error.message
    _nothing_stored(sha256)


def test_a_link_resolves_through_the_directorys_own_manifest(tmp_path: Path) -> None:
    linked = _library(tmp_path / "dev", "tracks", "0.5")
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [_link(linked)])
    argv, said = _heard(QUERY.format(call="tracks.lib.quieter"), _packages(project))
    assert argv == _argv(_quieter("0.5") + QUERY.format(call="quieter"))
    assert _codes(said) == [WarningCode.LINKED_PACKAGE]


def test_a_link_picks_up_an_edit_made_after_the_lockfile_was_written(tmp_path: Path) -> None:
    linked = _library(tmp_path / "dev", "tracks", "0.5")
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [_link(linked)])
    before, _ = _heard(QUERY.format(call="tracks.lib.quieter"), _packages(project))
    (linked / "src" / "lib.sql").write_text(_quieter("0.25"), encoding="utf-8")
    after, _ = _heard(QUERY.format(call="tracks.lib.quieter"), _packages(project))
    assert "volume=volume=0.5" in " ".join(before)
    assert "volume=volume=0.25" in " ".join(after)


def test_a_link_by_relative_path_resolves_against_the_lockfile(tmp_path: Path) -> None:
    _library(tmp_path / "dev", "tracks", "0.5")
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [{"kind": "link", "path": "../dev"}])
    argv, _ = _heard(QUERY.format(call="tracks.lib.quieter"), _packages(project))
    assert "volume=volume=0.5" in " ".join(argv)


def test_a_link_warns_once_however_many_call_sites(tmp_path: Path) -> None:
    linked = _library(tmp_path / "dev", "tracks", "0.5")
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [_link(linked)])
    _compiled, said = _heard(
        "COPY (SELECT tracks.lib.quieter(f.audio[1]), tracks.lib.quieter(f.audio[2]) "
        "FROM input('film.mkv') f) TO 'out.mkv'",
        _packages(project),
    )
    assert _codes(said) == [WarningCode.LINKED_PACKAGE]
    assert said[0].package == "tracks/lib"
    assert str(linked) in said[0].message


def test_a_linked_directory_with_no_manifest_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "dev"
    empty.mkdir()
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [_link(empty)])
    _refuses(project, "holds no ffrwd.json")


def test_a_linked_package_may_rename_itself_without_a_re_link(tmp_path: Path) -> None:
    """The entry records only the directory; the name is the manifest's."""
    linked = _library(tmp_path / "dev", "tracks", "0.5")
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [_link(linked)])
    assert "tracks/lib" in _packages(project).names()
    renamed = json.loads((linked / "ffrwd.json").read_text(encoding="utf-8"))
    renamed["name"] = "broadcast/audio"
    (linked / "ffrwd.json").write_text(json.dumps(renamed) + "\n", encoding="utf-8")
    packages = _packages(project)
    assert "broadcast/audio" in packages.names()
    argv, _ = _heard(QUERY.format(call="broadcast.audio.quieter"), packages)
    assert "volume=volume=0.5" in " ".join(argv)


def test_the_manifest_wins_over_a_lockfile_naming_its_package(tmp_path: Path) -> None:
    linked = _library(tmp_path / "dev", "me", "0.25", package="edits")
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": _quieter("0.5")})
    _lock(project, [_link(linked)])
    argv, said = _heard(QUERY.format(call="me.edits.quieter"), _packages(project))
    assert "volume=volume=0.5" in " ".join(argv)
    # The link lost the claim, so nothing resolves in it and nothing warns.
    assert said == []


def test_the_local_lockfile_wins_over_the_global_one(store_home: Path, tmp_path: Path) -> None:
    _lock(store_home, [_installed(_library(tmp_path / "far", "tracks", "0.25"))])
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [_installed(_library(tmp_path / "near", "tracks", "0.5"))])
    argv, said = _heard(QUERY.format(call="tracks.lib.quieter"), _packages(project))
    assert "volume=volume=0.5" in " ".join(argv)
    assert said == []


def test_all_three_layers_answer_in_order(store_home: Path, tmp_path: Path) -> None:
    _lock(
        store_home,
        [
            _installed(_library(tmp_path / "g-shadowed", "me", "0.1", package="edits")),
            _installed(_library(tmp_path / "g-tracks", "tracks", "0.2")),
            _installed(_library(tmp_path / "g-only", "far", "0.3")),
        ],
    )
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": _quieter("0.9")})
    _lock(project, [_installed(_library(tmp_path / "l-tracks", "tracks", "0.8"))])
    packages = _packages(project)
    assert packages.names() == ("far/lib", "me/edits", "tracks/lib")
    assert packages.namespaces() == ("far", "me", "tracks")
    assert [packages.packages[name].layer for name in packages.names()] == [
        "global",
        "project",
        "local",
    ]
    graph = " ".join(
        _heard(
            "COPY (SELECT me.edits.quieter(f.audio[1]), tracks.lib.quieter(f.audio[2]), "
            "far.lib.quieter(f.audio[3]) FROM input('film.mkv') f) TO 'out.mkv'",
            packages,
        )[0]
    )
    assert "volume=volume=0.9" in graph
    assert "volume=volume=0.8" in graph
    assert "volume=volume=0.3" in graph


def test_landing_on_the_global_layer_inside_a_project_warns(
    store_home: Path, tmp_path: Path
) -> None:
    _lock(store_home, [_installed(_library(tmp_path / "far", "tracks", "0.5"))])
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _compiled, said = _heard(QUERY.format(call="tracks.lib.quieter"), _packages(project))
    assert _codes(said) == [WarningCode.GLOBAL_PACKAGE]
    assert said[0].package == "tracks/lib"
    assert "tracks/lib" in (said[0].hint or "")


def test_a_global_package_outside_a_project_has_nothing_to_warn_about(
    store_home: Path, tmp_path: Path
) -> None:
    _lock(store_home, [_installed(_library(tmp_path / "far", "tracks", "0.5"))])
    bare = tmp_path / "bare"
    bare.mkdir()
    packages = discover(bare)
    assert packages is not None and not packages.in_project
    argv, said = _heard(QUERY.format(call="tracks.lib.quieter"), packages)
    assert "volume=volume=0.5" in " ".join(argv)
    assert said == []


def test_a_global_link_warns_about_both(store_home: Path, tmp_path: Path) -> None:
    linked = _library(tmp_path / "dev", "tracks", "0.5")
    _lock(store_home, [_link(linked)])
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _compiled, said = _heard(QUERY.format(call="tracks.lib.quieter"), _packages(project))
    assert set(_codes(said)) == {WarningCode.LINKED_PACKAGE, WarningCode.GLOBAL_PACKAGE}


def test_a_package_nothing_calls_is_never_warned_about(store_home: Path, tmp_path: Path) -> None:
    _lock(store_home, [_installed(_library(tmp_path / "far", "tracks", "0.5"))])
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": _quieter("0.5")})
    _compiled, said = _heard(QUERY.format(call="me.edits.quieter"), _packages(project))
    assert said == []


def test_a_lockfile_alone_is_a_project(store_home: Path, tmp_path: Path) -> None:
    _lock(store_home, [_installed(_library(tmp_path / "far", "tracks", "0.5"))])
    work = tmp_path / "work"
    _lock(work, [_installed(_library(tmp_path / "near", "own", "0.5"))])
    packages = discover(work)
    assert packages is not None and packages.in_project
    assert packages.names() == ("own/lib", "tracks/lib")


def test_nothing_anywhere_is_still_no_project(store_home: Path, tmp_path: Path) -> None:
    bare = tmp_path / "bare"
    bare.mkdir()
    assert discover(bare) is None


# ---------------------------------------------------------------------------
# the version binding is per-manifest: a call resolves at the version the
# CALLING package (or the project itself) depends on, never a flat table
# ---------------------------------------------------------------------------


def test_the_set_carries_the_projects_own_wants(store_home: Path, tmp_path: Path) -> None:
    entry = _installed(_library(tmp_path / "built", "broadcast", "0.5", package="tracks"))
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [entry], dependencies={"broadcast/tracks": "1.0.0"})
    packages = _packages(project)
    assert packages.wants[packages.project or ""] == {"broadcast/tracks": "1.0.0"}
    found = packages.resolve(None, "broadcast/tracks")
    assert found is not None and found.name == "broadcast/tracks"
    assert packages.resolve(None, "nothing/here") is None


def test_a_want_naming_a_version_nothing_installed_falls_back_to_the_canonical_entry(
    store_home: Path, tmp_path: Path
) -> None:
    """A lockfile's own `dependencies` naming a version no entry pins is stale, not a crash."""
    entry = _installed(_library(tmp_path / "built", "broadcast", "0.5", package="tracks"))
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [entry], dependencies={"broadcast/tracks": "9.9.9"})
    packages = _packages(project)
    found = packages.resolve(None, "broadcast/tracks")
    assert found is not None and found.version == "1.0.0"


def _tool(root: Path, wants: str) -> Path:
    """A package whose RECIPE calls into another package at `wants`."""
    (root / "queries").mkdir(parents=True, exist_ok=True)
    recipe = [
        "-- Quieten the first track.",
        "-- variables: dest (output path)",
        "-- example: ffrwd compile -f queries/go.sql -v dest=out.mkv",
        "COPY (SELECT shared.d.quieter(f.audio[1]) FROM input('film.mkv') f) TO :'dest'",
    ]
    _write(root / "queries" / "go.sql", chr(10).join(recipe) + chr(10))
    manifest = {
        "name": "me/tool",
        "version": "1.0.0",
        "bin": {"go": "queries/go.sql"},
        "dependencies": {"shared/d": wants},
    }
    _write(root / "ffrwd.json", json.dumps(manifest, indent=2) + chr(10))
    return root


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_a_recipe_reaches_the_version_its_own_package_declares(
    store_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A recipe is its package's source, not anonymous text.

    The project binds shared/d at 2.0.0; the package whose recipe runs binds
    1.0.0. The recipe must get the version it was written against.
    """
    d_low = _installed(_library(tmp_path / "d1", "shared", "0.1", package="d", version="1.0.0"))
    d_high = _installed(_library(tmp_path / "d2", "shared", "0.2", package="d", version="2.0.0"))
    tool = _installed(_tool(tmp_path / "tool", "1.0.0"), dependencies={"shared/d": "1.0.0"})
    project = tmp_path / "work"
    _project(project, files={}, manifest={"name": "other/edits"})
    _lock(project, [d_low, d_high, tool], dependencies={"shared/d": "2.0.0"})

    code, out, err = _run(
        project, monkeypatch, capsys, "compile", "me.tool.go", "-v", "dest=o.mkv"
    )
    assert code == 0, err
    assert "volume=volume=0.1" in out, out

def test_two_dependents_resolve_a_shared_name_at_their_own_version(
    store_home: Path, tmp_path: Path
) -> None:
    """B depends on D@1, C depends on D@2; each keeps calling into its own."""
    reach = (
        "CREATE FUNCTION use(track audio_stream) RETURNS audio_stream AS $$\n"
        "  SELECT shared.d.quieter(track)\n"
        "$$ LANGUAGE sql;\n"
    )
    d_low = _installed(_library(tmp_path / "d1", "shared", "0.1", package="d", version="1.0.0"))
    d_high = _installed(_library(tmp_path / "d2", "shared", "0.2", package="d", version="2.0.0"))
    b = _installed(
        _library(
            tmp_path / "b", "me", "", package="b", member="use", src=reach,
            dependencies={"shared/d": "1.0.0"},
        ),
        dependencies={"shared/d": "1.0.0"},
    )
    c = _installed(
        _library(
            tmp_path / "c", "me", "", package="c", member="use", src=reach,
            dependencies={"shared/d": "2.0.0"},
        ),
        dependencies={"shared/d": "2.0.0"},
    )
    project = tmp_path / "work"
    _project(project, files={}, manifest={"name": "other/edits"})
    _lock(project, [d_low, d_high, b, c])
    packages = _packages(project)
    through_b = _argv(QUERY.format(call="me.b.use"), packages)
    through_c = _argv(QUERY.format(call="me.c.use"), packages)
    assert "volume=volume=0.1" in " ".join(through_b)
    assert "volume=volume=0.2" in " ".join(through_c)


def test_two_dependents_in_one_query_each_inline_their_own_version(
    store_home: Path, tmp_path: Path
) -> None:
    """The headline case: ONE compile inlines B and C, each resolving `shared.d` on its own.

    B and C share a package NAME for their shared dependency ('shared/d'), so
    this is only decidable if the scope a bare call resolves in -- and the
    cache of what a package's lib files define -- is kept per (name, version),
    never conflated by name alone once two versions of it are both read in
    the same compile.
    """
    reach = (
        "CREATE FUNCTION use(track audio_stream) RETURNS audio_stream AS $$\n"
        "  SELECT shared.d.quieter(track)\n"
        "$$ LANGUAGE sql;\n"
    )
    d_low = _installed(_library(tmp_path / "d1", "shared", "0.1", package="d", version="1.0.0"))
    d_high = _installed(_library(tmp_path / "d2", "shared", "0.2", package="d", version="2.0.0"))
    b = _installed(
        _library(
            tmp_path / "b", "me", "", package="b", member="use", src=reach,
            dependencies={"shared/d": "1.0.0"},
        ),
        dependencies={"shared/d": "1.0.0"},
    )
    c = _installed(
        _library(
            tmp_path / "c", "me", "", package="c", member="use", src=reach,
            dependencies={"shared/d": "2.0.0"},
        ),
        dependencies={"shared/d": "2.0.0"},
    )
    project = tmp_path / "work"
    _project(project, files={}, manifest={"name": "other/edits"})
    _lock(project, [d_low, d_high, b, c])
    argv = _argv(
        "COPY (SELECT me.b.use(f.audio[1]), me.c.use(f.audio[2]) "
        "FROM input('film.mkv') f) TO 'out.mkv'",
        _packages(project),
    )
    graph = " ".join(argv)
    assert "volume=volume=0.1" in graph
    assert "volume=volume=0.2" in graph


# ---------------------------------------------------------------------------
# a malformed lockfile is a typed rejection, like a malformed manifest
# ---------------------------------------------------------------------------


def test_a_lockfile_that_is_not_json_is_anchored(tmp_path: Path) -> None:
    _project(tmp_path, files={"src/own.sql": NORMALIZE})
    _lock(tmp_path, [], text="{ nope\n")
    error = _refuses(tmp_path, "ffrwd.lock")
    assert error.code is ErrorCode.UNSUPPORTED_SQL
    assert error.line == 1


def test_a_lockfile_from_another_format_version_is_refused(tmp_path: Path) -> None:
    _project(tmp_path, files={"src/own.sql": NORMALIZE})
    _lock(tmp_path, [], text='{"format_version": 1, "reproducible": true, "packages": []}\n')
    _refuses(tmp_path, "lockfile format 1")


@pytest.mark.parametrize("missing", ["format_version", "reproducible", "packages"])
def test_every_lockfile_key_is_required(tmp_path: Path, missing: str) -> None:
    _project(tmp_path, files={"src/own.sql": NORMALIZE})
    written = {"format_version": LOCK_FORMAT_VERSION, "reproducible": True, "packages": []}
    del written[missing]
    _lock(tmp_path, [], text=json.dumps(written))
    _refuses(tmp_path, f'is missing "{missing}"')


def test_a_lockfile_claiming_to_be_reproducible_while_linking_is_refused(tmp_path: Path) -> None:
    linked = _library(tmp_path / "dev", "tracks", "0.5")
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [_link(linked)], reproducible=True)
    _refuses(project, "claims to be reproducible")


def test_a_lockfile_says_in_its_own_text_why_it_is_not_reproducible(tmp_path: Path) -> None:
    linked = _library(tmp_path / "dev", "tracks", "0.5")
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    written = _lock(project, [_link(linked)]).read_text(encoding="utf-8")
    assert '"reproducible": false' in written
    assert "not_reproducible_because" in written


def test_two_entries_naming_one_package_are_refused(tmp_path: Path) -> None:
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    entry = {
        "kind": "registry",
        "name": "broadcast/tracks",
        "version": "1.0.0",
        "sha256": "a" * 64,
        "store": store.entry_path("a" * 64),
    }
    _lock(project, [entry, dict(entry)])
    _refuses(project, "two entries pin package 'broadcast/tracks' at version '1.0.0'")


def test_a_registry_entry_and_a_link_at_one_version_are_refused(tmp_path: Path) -> None:
    """Same name AND same version, one from each kind: still one identity, still refused."""
    linked = _library(tmp_path / "dev", "broadcast", "0.5", package="tracks")
    entry = _installed(_library(tmp_path / "built", "broadcast", "0.25", package="tracks"))
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [entry, _link(linked)])
    _refuses(project, "two entries name package 'broadcast/tracks' 1.0.0")


def test_a_registry_entry_and_a_link_at_different_versions_coexist(tmp_path: Path) -> None:
    """A different version of the same name is not a collision -- both simply resolve."""
    linked = _library(tmp_path / "dev", "broadcast", "0.5", package="tracks", version="2.0.0")
    entry = _installed(_library(tmp_path / "built", "broadcast", "0.25", package="tracks"))
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [entry, _link(linked)])
    packages = _packages(project)
    assert packages.versions["broadcast/tracks"].keys() == {"1.0.0", "2.0.0"}


def test_an_entry_of_no_known_kind_is_refused(tmp_path: Path) -> None:
    _project(tmp_path, files={"src/own.sql": NORMALIZE})
    _lock(tmp_path, [{"path": "../dev"}])
    _refuses(tmp_path, 'a package entry has no "kind"')


def test_an_entry_missing_a_key_names_it(tmp_path: Path) -> None:
    _project(tmp_path, files={"src/own.sql": NORMALIZE})
    _lock(tmp_path, [{"kind": "link"}])
    _refuses(tmp_path, 'a link entry is missing "path"')


def test_an_unknown_entry_key_gets_a_did_you_mean(tmp_path: Path) -> None:
    _project(tmp_path, files={"src/own.sql": NORMALIZE})
    _lock(tmp_path, [{"kind": "link", "path": "../dev", "pth": "x"}])
    error = _refuses(tmp_path, "unknown key 'pth'")
    assert "did you mean 'path'?" in (error.hint or "")


def test_a_namespace_key_is_no_longer_part_of_an_entry(tmp_path: Path) -> None:
    _project(tmp_path, files={"src/own.sql": NORMALIZE})
    _lock(tmp_path, [{"kind": "link", "namespace": "tracks", "path": "../dev"}])
    _refuses(tmp_path, "unknown key 'namespace'")


@pytest.mark.parametrize(
    ("claimed", "needle"),
    [
        ("ffmpeg/tracks", "namespace 'ffmpeg' is reserved"),
        ("ffrwd/speed", "is named for the macro ffrwd.speed()"),
    ],
)
def test_a_name_ffrwd_keeps_in_a_lockfile_is_refused(
    tmp_path: Path, claimed: str, needle: str
) -> None:
    _project(tmp_path, files={"src/own.sql": NORMALIZE})
    _lock(
        tmp_path,
        [
            {
                "kind": "registry",
                "name": claimed,
                "version": "1.0.0",
                "sha256": "a" * 64,
                "store": store.entry_path("a" * 64),
            }
        ],
    )
    _refuses(tmp_path, needle)


def test_a_rejection_points_at_the_entry_it_is_about(tmp_path: Path) -> None:
    first = _library(tmp_path / "one", "good", "0.5")
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    path = _lock(project, [_link(first), {"kind": "link", "path": "../later", "pth": "x"}])
    error = _refuses(project, "unknown key 'pth'")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert error.line is not None
    assert '"../later"' in lines[error.line - 1]


# ---------------------------------------------------------------------------
# the diagnostic channel reaches the CLI's stderr and the MCP tool result
# ---------------------------------------------------------------------------


def test_the_cli_prints_the_warning_on_stderr_and_the_command_on_stdout(
    store_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _lock(store_home, [_installed(_library(tmp_path / "far", "tracks", "0.5"))])
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    monkeypatch.chdir(project)
    assert cli.main(["compile", QUERY.format(call="tracks.lib.quieter")]) == 0
    captured = capsys.readouterr()
    assert "volume=volume=0.5" in captured.out
    assert "warning:" not in captured.out
    assert "warning: package 'tracks/lib' was resolved from the machine-wide" in captured.err
    assert "hint:" in captured.err


def test_the_cli_says_it_once_though_it_compiles_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    linked = _library(tmp_path / "dev", "tracks", "0.5")
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [_link(linked)])
    monkeypatch.chdir(project)
    # A bare SELECT: `compile` refuses it, then tries the table fallback, so
    # the same text compiles twice in one command.
    assert cli.main(["compile", "SELECT tracks.lib.quieter(f.audio[1]) FROM input('f.mkv') f"]) == 2
    assert capsys.readouterr().err.count("warning:") == 1


def test_validate_keeps_the_warning_off_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    linked = _library(tmp_path / "dev", "tracks", "0.5")
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [_link(linked)])
    monkeypatch.chdir(project)
    assert cli.main(["validate", QUERY.format(call="tracks.lib.quieter")]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "warning: package 'tracks/lib' is linked to" in captured.err


def test_the_mcp_compile_tool_returns_the_warnings(store_home: Path, tmp_path: Path) -> None:
    _lock(store_home, [_installed(_library(tmp_path / "far", "tracks", "0.5"))])
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    result = mcp_tools.compile_query(QUERY.format(call="tracks.lib.quieter"), None, str(project))
    assert [w["code"] for w in result["warnings"]] == [WarningCode.GLOBAL_PACKAGE.value]
    assert result["warnings"][0]["package"] == "tracks/lib"


def test_the_mcp_validate_tool_answers_with_warnings_and_no_code(tmp_path: Path) -> None:
    linked = _library(tmp_path / "dev", "tracks", "0.5")
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [_link(linked)])
    result = mcp_tools.validate_query(QUERY.format(call="tracks.lib.quieter"), None, str(project))
    assert "code" not in result
    assert [w["code"] for w in result["warnings"]] == [WarningCode.LINKED_PACKAGE.value]


def test_the_mcp_tools_stay_silent_with_nothing_to_say(tmp_path: Path) -> None:
    _project(tmp_path, files={"src/own.sql": _quieter("0.5")})
    query = QUERY.format(call="me.edits.quieter")
    assert mcp_tools.validate_query(query, None, str(tmp_path)) == {}
    assert mcp_tools.compile_query(query, None, str(tmp_path))["warnings"] == []


# ---------------------------------------------------------------------------
# `ffrwd list`: what the project and its dependencies provide
# ---------------------------------------------------------------------------


def _list(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *flags: str,
) -> tuple[int, str, str]:
    """Run `ffrwd list` with `root` as the working directory."""
    monkeypatch.chdir(root)
    code = cli.main(["list", *flags])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_list_prints_the_exports_recipes_and_dependencies_a_project_provides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _project(
        tmp_path,
        files={"src/tracks.sql": QUIETER + PICK, "queries/split.sql": RECIPE},
        manifest={**_BIN, "dependencies": {"broadcast/tracks": "^1.2.0"}},
    )
    code, out, _err = _list(tmp_path, monkeypatch, capsys)
    assert code == 0
    assert "quieter(track audio_stream, factor number) | audio_stream" in out
    assert "pick(path text)" in out and "TABLE(track audio_stream)" in out
    assert "split-chapters" in out
    assert "source (input media path), dest (output path)" in out  # required column
    assert "queries/split.sql" in out
    assert "me/edits | 0.1.0   | project | false" in out
    assert "broadcast/tracks" in out and "^1.2.0" in out


def test_list_outside_a_project_prints_empty_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bare = tmp_path / "no_project"
    bare.mkdir()
    code, out, _err = _list(bare, monkeypatch, capsys)
    assert code == 0
    assert out.count("(0 rows)") == 4


def test_list_as_json_carries_the_signatures_and_the_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _project(
        tmp_path,
        files={"src/tracks.sql": QUIETER, "queries/split.sql": RECIPE},
        manifest={**_BIN, "dependencies": {"broadcast/tracks": "^1.2.0"}},
    )
    code, out, _err = _list(tmp_path, monkeypatch, capsys, "--json")
    assert code == 0
    listed = json.loads(out)["packages"]
    assert [package["name"] for package in listed] == ["me/edits"]
    package = listed[0]
    assert package["layer"] == "project"
    assert package["linked"] is False
    assert package["exports"] == [
        {
            "name": "quieter",
            "params": [
                {"name": "track", "type": "audio_stream", "default": None},
                {"name": "factor", "type": "number", "default": None},
            ],
            "returns": "audio_stream",
            "file": "src/tracks.sql",
        }
    ]
    assert package["recipes"] == [
        {
            "name": "split-chapters",
            "file": "queries/split.sql",
            "required": [
                {"name": "source", "description": "input media path"},
                {"name": "dest", "description": "output path"},
            ],
            "optional": [],
        }
    ]
    assert package["dependencies"] == [{"name": "broadcast/tracks", "range": "^1.2.0"}]


def test_list_names_the_layer_and_marks_a_linked_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    linked = _library(tmp_path / "dev", "tracks", "0.5")
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [_link(linked)])
    code, out, _err = _list(project, monkeypatch, capsys, "--json")
    assert code == 0
    listed = {package["name"]: package for package in json.loads(out)["packages"]}
    assert listed["me/edits"]["layer"] == "project" and listed["me/edits"]["linked"] is False
    assert listed["tracks/lib"]["layer"] == "local" and listed["tracks/lib"]["linked"] is True
    assert [f["name"] for f in listed["tracks/lib"]["exports"]] == ["quieter"]


def test_list_reports_a_malformed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _project(tmp_path, files={"src/tracks.sql": QUIETER}, text="{ nope\n")
    code, out, err = _list(tmp_path, monkeypatch, capsys)
    assert code == 1
    assert out == ""
    assert "ffrwd.json" in err


def test_list_reports_a_lib_file_that_is_not_a_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _project(tmp_path, files={"src/tracks.sql": QUIETER + "SELECT 1;"})
    code, _out, err = _list(tmp_path, monkeypatch, capsys)
    assert code == 1
    assert "is not a CREATE FUNCTION" in err


def test_list_takes_no_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as caught:
        cli.main(["list", "SELECT 1"])
    assert caught.value.code == 2


# ---------------------------------------------------------------------------
# writing the two files
# ---------------------------------------------------------------------------


def _registry_entry(name: str = "broadcast/tracks") -> RegistryEntry:
    sha256 = "a" * 64
    return RegistryEntry(
        name=name,
        version="1.2.0",
        sha256=sha256,
        store=store.entry_path(sha256),
    )


def test_a_written_lockfile_reads_back_as_what_was_written(tmp_path: Path) -> None:
    path = tmp_path / "ffrwd.lock"
    entries = (_registry_entry(), LinkEntry(path="../my-lib"))
    write_lockfile(path, entries)
    assert read_lockfile(path).entries == entries


def test_writing_a_lockfile_twice_writes_the_same_bytes(tmp_path: Path) -> None:
    path = tmp_path / "ffrwd.lock"
    entries = (_registry_entry(), LinkEntry(path="../my-lib"))
    write_lockfile(path, entries)
    first = path.read_bytes()
    write_lockfile(path, read_lockfile(path).entries)
    assert path.read_bytes() == first
    assert read_lockfile(path).entries == entries


def test_a_written_lockfile_is_lf_only(tmp_path: Path) -> None:
    path = tmp_path / "ffrwd.lock"
    write_lockfile(path, (_registry_entry(),))
    data = path.read_bytes()
    assert b"\r" not in data
    assert data.endswith(b"\n")


def test_a_lockfile_of_registry_entries_claims_to_be_reproducible(tmp_path: Path) -> None:
    path = tmp_path / "ffrwd.lock"
    write_lockfile(path, (_registry_entry(),))
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["format_version"] == LOCK_FORMAT_VERSION
    assert written["reproducible"] is True
    assert "not_reproducible_because" not in written
    assert read_lockfile(path).reproducible is True


def test_a_lockfile_holding_a_link_says_it_is_not_reproducible_and_why(tmp_path: Path) -> None:
    path = tmp_path / "ffrwd.lock"
    write_lockfile(path, (_registry_entry(), LinkEntry(path="../my-lib")))
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["reproducible"] is False
    assert "linked to a working directory" in written["not_reproducible_because"]
    assert read_lockfile(path).reproducible is False


def test_an_empty_lockfile_reads_back(tmp_path: Path) -> None:
    path = tmp_path / "ffrwd.lock"
    write_lockfile(path, ())
    lock = read_lockfile(path)
    assert lock.entries == () and lock.reproducible is True


def test_the_written_entry_order_is_the_order_given(tmp_path: Path) -> None:
    path = tmp_path / "ffrwd.lock"
    entries = (_registry_entry("zulu/lib"), _registry_entry("alpha/lib"))
    write_lockfile(path, entries)
    assert read_lockfile(path).entries == entries


def test_with_entry_replaces_and_without_entry_removes() -> None:
    held = (_registry_entry("broadcast/tracks"), _registry_entry("far/other"))
    link = LinkEntry(path="../dev")
    assert with_entry(held, link, held[0]) == (link, held[1])
    appended = with_entry(held, _registry_entry("new/lib"))
    assert appended[-1] == _registry_entry("new/lib")
    assert without_entry(held, held[0]) == (held[1],)
    assert without_entry(held, link) == held


def test_a_written_manifest_reads_back_as_the_package_it_declares(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.sql").write_text(QUIETER, encoding="utf-8")
    (tmp_path / "queries").mkdir()
    (tmp_path / "queries" / "split.sql").write_text(RECIPE, encoding="utf-8")
    path = tmp_path / "ffrwd.json"
    write_manifest(
        path,
        name="me/edits",
        version="0.1.0",
        description="what it is",
        lib={"quieter": "src/lib.sql"},
        bin={"split-chapters": "queries/split.sql"},
        dependencies={"broadcast/tracks": "^1.2.0"},
    )
    package = read_manifest(path)
    assert package.name == "me/edits" and package.version == "0.1.0"
    assert list(package.exports) == ["quieter"]
    assert list(package.recipes) == ["split-chapters"]
    assert package.dependencies == {"broadcast/tracks": "^1.2.0"}
    assert b"\r" not in path.read_bytes()


def test_a_written_manifest_takes_the_string_form_of_lib_and_bin(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.sql").write_text(QUIETER.replace("quieter", "edits"), encoding="utf-8")
    (tmp_path / "queries").mkdir()
    (tmp_path / "queries" / "split.sql").write_text(RECIPE, encoding="utf-8")
    path = tmp_path / "ffrwd.json"
    write_manifest(
        path, name="me/edits", version="0.1.0", lib="src/lib.sql", bin="queries/split.sql"
    )
    package = read_manifest(path)
    assert list(package.exports) == ["edits"]
    assert list(package.recipes) == ["edits"]
    assert package.export() == tmp_path / "src" / "lib.sql"
    assert package.recipe() == tmp_path / "queries" / "split.sql"


def test_a_manifest_leaves_out_what_it_was_not_given(tmp_path: Path) -> None:
    path = tmp_path / "ffrwd.json"
    write_manifest(path, name="me/edits", version="0.1.0")
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "name": "me/edits",
        "version": "0.1.0",
    }
    assert dict(read_manifest(path).exports) == {}


def test_a_write_into_a_directory_that_is_a_file_is_a_rejection(tmp_path: Path) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory\n", encoding="utf-8")
    with pytest.raises(FfrwdError) as caught:
        write_lockfile(blocked / "ffrwd.lock", ())
    assert "could not be written" in caught.value.message


# ---------------------------------------------------------------------------
# `ffrwd init`
# ---------------------------------------------------------------------------


def _run(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *argv: str,
) -> tuple[int, str, str]:
    """Run the CLI with `root` as the working directory."""
    monkeypatch.chdir(root)
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_init_writes_a_project_that_reads_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "my-edits"
    root.mkdir()
    code, out, _err = _run(root, monkeypatch, capsys, "init", "--namespace", "me")
    assert code == 0
    assert "ffrwd.json" in out
    assert "--namespace" in out
    package = read_manifest(root / "ffrwd.json")
    assert package.name == "me/my_edits" and package.version == "0.1.0"
    assert list(package.recipes) == ["resize"]
    assert read_lockfile(root / "ffrwd.lock").entries == ()
    assert "-- variables:" in (root / "recipes" / "resize.sql").read_text(encoding="utf-8")


def test_init_then_list_then_run_the_starter_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "my-edits"
    root.mkdir()
    assert _run(root, monkeypatch, capsys, "init", "--namespace", "me")[0] == 0

    code, out, _err = _run(root, monkeypatch, capsys, "list")
    assert code == 0
    assert "resize" in out and "me/my_edits" in out

    code, out, _err = _run(
        root, monkeypatch, capsys, "compile", "resize", "-v", "source=in.mp4", "-v", "dest=out.mp4"
    )
    assert code == 0
    assert out.startswith("ffmpeg -i in.mp4 ")
    assert "scale=width=-2:height=720" in out


def test_init_takes_the_whole_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "whatever"
    root.mkdir()
    code, out, _err = _run(root, monkeypatch, capsys, "init", "--name", "broadcast/tracks")
    assert code == 0
    assert "--name" in out
    assert read_manifest(root / "ffrwd.json").name == "broadcast/tracks"


def test_init_derives_the_namespace_from_the_git_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    root = tmp_path / "tracks"
    root.mkdir()
    subprocess.run([git, "init", "-q"], cwd=root, check=True)
    subprocess.run(
        [git, "remote", "add", "origin", "https://github.com/broadcast/tracks.git"],
        cwd=root,
        check=True,
    )
    code, out, _err = _run(root, monkeypatch, capsys, "init")
    assert code == 0
    assert "git remote" in out
    assert read_manifest(root / "ffrwd.json").name == "broadcast/tracks"


def test_init_without_a_derivable_namespace_requires_the_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "work"
    root.mkdir()
    code, _out, err = _run(root, monkeypatch, capsys, "init")
    assert code == 1
    assert "--namespace" in err
    assert not (root / "ffrwd.json").exists()
    assert not (root / "queries").exists()


def test_init_refuses_to_overwrite_a_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _project(tmp_path, files={"src/own.sql": NORMALIZE})
    before = (tmp_path / "ffrwd.json").read_bytes()
    code, _out, err = _run(tmp_path, monkeypatch, capsys, "init", "--namespace", "me")
    assert code == 1
    assert "ffrwd.json already exists" in err
    assert (tmp_path / "ffrwd.json").read_bytes() == before


def test_init_refuses_to_overwrite_a_lockfile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _lock(tmp_path, [])
    code, _out, err = _run(tmp_path, monkeypatch, capsys, "init", "--namespace", "me")
    assert code == 1
    assert "ffrwd.lock already exists" in err
    assert not (tmp_path / "ffrwd.json").exists()


@pytest.mark.parametrize("name", ["9lives", "..."])
def test_init_refuses_a_name_no_package_segment_comes_out_of(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], name: str
) -> None:
    root = tmp_path / "work"
    root.mkdir()
    code, _out, err = _run(
        root, monkeypatch, capsys, "init", "--name", name, "--namespace", "me"
    )
    assert code == 1
    assert "gives no package segment" in err
    assert not (root / "ffrwd.json").exists()
    assert not (root / "queries").exists()


def test_init_refuses_a_namespace_that_is_not_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "work"
    root.mkdir()
    code, _out, err = _run(root, monkeypatch, capsys, "init", "--namespace", "Not One")
    assert code == 1
    assert "--namespace gives no namespace" in err


def test_init_refuses_a_reserved_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "work"
    root.mkdir()
    code, _out, err = _run(root, monkeypatch, capsys, "init", "--namespace", "ffmpeg")
    assert code == 1
    assert "reserved" in err


def test_init_rust_writes_a_module_package_that_reads_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "my-filter"
    root.mkdir()
    code, out, _err = _run(root, monkeypatch, capsys, "init", "--rust", "--namespace", "me")
    assert code == 0
    assert "cargo build --target wasm32-wasip2 --release" in out

    package = read_manifest(root / "ffrwd.json")
    assert package.name == "me/my_filter" and package.version == "0.1.0"
    assert list(package.exports) == ["invert"] and list(package.recipes) == ["invert"]
    assert dict(package.dependencies) == {"ffrwd/wasm": wasm.WORLD_VERSION}
    # Declared empty rather than absent: the scaffold shows its author where
    # they go.
    assert package.keywords == () and package.capabilities == ()
    assert read_lockfile(root / "ffrwd.lock").entries == ()
    written = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert written == {
        ".ffrwdignore",
        ".gitignore",
        "Cargo.toml",
        "README.md",
        "build.rs",
        "ffrwd.json",
        "ffrwd.lock",
        "recipes/invert.sql",
        "src/invert.sql",
        "src/lib.rs",
    }


def test_init_rust_wires_the_crate_to_the_sql_that_names_its_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The crate's name, the path the lib SQL names and the call the recipe
    writes are three spellings of one thing; a scaffold that got any of them
    wrong would not compile."""
    root = tmp_path / "my-filter"
    root.mkdir()
    assert _run(root, monkeypatch, capsys, "init", "--rust", "--namespace", "me")[0] == 0

    def _text(name: str) -> str:
        return (root / name).read_text(encoding="utf-8")

    assert 'name = "my_filter"' in _text("Cargo.toml")
    assert "target/wasm32-wasip2/release/my_filter.wasm" in _text("src/invert.sql")
    assert "LANGUAGE wasm" in _text("src/invert.sql")
    assert "me.my_filter.invert(" in _text("recipes/invert.sql")
    assert "-- variables:" in _text("recipes/invert.sql")
    assert "-- example: ffrwd run invert" in _text("recipes/invert.sql")
    # The wit comes from the environment or from the installed package, and
    # lands where the bindings macro reads it.
    assert "FFRWD_WIT_DIR" in _text("build.rs")
    assert 'args(["path", WIT_PACKAGE])' in _text("build.rs")
    assert 'path: "wit"' in _text("src/lib.rs")
    assert _text(".gitignore").split() == ["target/", "wit/"]


def test_init_rust_refuses_to_overwrite_any_file_it_would_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "my-filter"
    root.mkdir()
    (root / "Cargo.toml").write_text("# mine\n", encoding="utf-8")
    code, _out, err = _run(root, monkeypatch, capsys, "init", "--rust", "--namespace", "me")
    assert code == 1
    assert "Cargo.toml already exists" in err
    assert not (root / "ffrwd.json").exists()


# ---------------------------------------------------------------------------
# `ffrwd link` and `ffrwd unlink`
# ---------------------------------------------------------------------------


def test_link_records_the_directory_and_names_the_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _library(tmp_path / "dev", "tracks", "0.5")
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [])
    code, out, _err = _run(project, monkeypatch, capsys, "link", "../dev")
    assert code == 0
    assert "linked tracks/lib -> ../dev" in out
    lock = read_lockfile(project / "ffrwd.lock")
    assert lock.entries == (LinkEntry(path="../dev"),)
    assert lock.reproducible is False


def test_a_linked_package_is_callable_right_after_linking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _library(tmp_path / "dev", "tracks", "0.5")
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [])
    assert _run(project, monkeypatch, capsys, "link", "../dev")[0] == 0
    code, out, err = _run(
        project, monkeypatch, capsys, "compile", QUERY.format(call="tracks.lib.quieter")
    )
    assert code == 0
    assert "volume=volume=0.5" in out
    assert "warning: package 'tracks/lib' is linked to" in err


def test_link_replaces_what_pinned_the_package(
    store_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [_installed(_library(tmp_path / "far", "tracks", "0.5"))])
    _library(tmp_path / "dev", "tracks", "0.9")
    code, out, _err = _run(project, monkeypatch, capsys, "link", "../dev")
    assert code == 0
    assert "replacing the installed tracks/lib 1.0.0" in out
    assert read_lockfile(project / "ffrwd.lock").entries == (LinkEntry(path="../dev"),)


def test_link_outside_a_project_names_both_ways_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _library(tmp_path / "dev", "tracks", "0.5")
    bare = tmp_path / "elsewhere"
    bare.mkdir()
    code, _out, err = _run(bare, monkeypatch, capsys, "link", "../dev")
    assert code == 2
    assert "ffrwd init" in err and "link -g" in err
    assert not (bare / "ffrwd.lock").exists()


def test_link_writes_the_machine_wide_lockfile(
    store_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    linked = _library(tmp_path / "dev", "tracks", "0.5")
    bare = tmp_path / "elsewhere"
    bare.mkdir()
    code, _out, _err = _run(bare, monkeypatch, capsys, "link", "-g", str(linked))
    assert code == 0
    lock = read_lockfile(store.global_lock_path())
    # Absolute, since the machine-wide lockfile lives under the cache directory.
    assert lock.entries == (LinkEntry(path=str(linked.resolve())),)


def test_link_refuses_a_directory_holding_no_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "dev").mkdir()
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [])
    code, _out, err = _run(project, monkeypatch, capsys, "link", "../dev")
    assert code == 1
    assert "holds no ffrwd.json" in err
    assert read_lockfile(project / "ffrwd.lock").entries == ()


def test_unlink_removes_the_entry_by_package_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    linked = _library(tmp_path / "dev", "tracks", "0.5")
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [_link(linked)])
    code, out, _err = _run(project, monkeypatch, capsys, "unlink", "tracks/lib")
    assert code == 0
    assert "unlinked 'tracks/lib'" in out
    lock = read_lockfile(project / "ffrwd.lock")
    assert lock.entries == () and lock.reproducible is True


def test_unlink_removes_a_dead_link_by_its_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A link whose directory lost its manifest has no name, and still goes away."""
    linked = _library(tmp_path / "dev", "tracks", "0.5")
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [_link(linked)])
    (linked / "ffrwd.json").unlink()
    code, _out, _err = _run(project, monkeypatch, capsys, "unlink", str(linked))
    assert code == 0
    assert read_lockfile(project / "ffrwd.lock").entries == ()


def test_unlink_names_what_is_linked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    linked = _library(tmp_path / "dev", "tracks", "0.5")
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [_link(linked)])
    code, _out, err = _run(project, monkeypatch, capsys, "unlink", "nope/nope")
    assert code == 1
    assert "nothing links 'nope/nope'" in err
    assert "hint: linked: tracks/lib" in err


def test_unlink_says_a_package_is_installed_rather_than_linked(
    store_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [_installed(_library(tmp_path / "far", "tracks", "0.5"))])
    code, _out, err = _run(project, monkeypatch, capsys, "unlink", "tracks/lib")
    assert code == 1
    assert "it is installed, not linked" in err
    assert len(read_lockfile(project / "ffrwd.lock").entries) == 1


def test_unlink_outside_a_project_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bare = tmp_path / "elsewhere"
    bare.mkdir()
    code, _out, err = _run(bare, monkeypatch, capsys, "unlink", "tracks/lib")
    assert code == 2
    assert "unlink -g" in err
    assert not (bare / "ffrwd.lock").exists()


# ---------------------------------------------------------------------------
# running a recipe by name
# ---------------------------------------------------------------------------

_MEDIA_QUERY = "COPY (SELECT scale(f.video[1], 640, 480) FROM input('x.mp4') f) TO 'out.mp4'"


@pytest.mark.parametrize("command", ["compile", "explain", "validate", "run"])
def test_every_subcommand_taking_a_query_takes_a_recipe_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    _project(
        tmp_path,
        files={"src/own.sql": NORMALIZE, "queries/split.sql": RECIPE},
        manifest=_BIN,
    )
    if command == "run":
        # The default tier executes no ffmpeg: reaching the check for one is
        # already proof the name resolved and the recipe compiled.
        monkeypatch.setattr(cli.binaries, "ffmpeg_path", lambda: None)
    code, _out, err = _run(
        tmp_path,
        monkeypatch,
        capsys,
        command,
        "split-chapters",
        "-v",
        "source=in.mkv",
        "-v",
        "dest=out.mkv",
    )
    if command == "run":
        assert code == 1 and "ffmpeg not found" in err
    else:
        assert code == 0, err


def test_the_default_recipe_is_reached_as_the_package_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _project(
        tmp_path,
        files={"queries/split.sql": RECIPE},
        manifest={"bin": "queries/split.sql"},
    )
    code, _out, err = _run(
        tmp_path,
        monkeypatch,
        capsys,
        "validate",
        "me.edits",
        "-v",
        "source=in.mkv",
        "-v",
        "dest=out.mkv",
    )
    assert code == 0, err


def test_a_qualified_recipe_name_runs_one_packages_bin_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`me.edits.split-chapters` names one entry of a map `bin`, for `compile` and `run` alike."""
    _project(
        tmp_path,
        files={"queries/split.sql": RECIPE},
        manifest=_BIN,
    )
    code, _out, err = _run(
        tmp_path,
        monkeypatch,
        capsys,
        "compile",
        "me.edits.split-chapters",
        "-v",
        "source=in.mkv",
        "-v",
        "dest=out.mkv",
    )
    assert code == 0, err

    monkeypatch.setattr(cli.binaries, "ffmpeg_path", lambda: None)
    code, _out, err = _run(
        tmp_path,
        monkeypatch,
        capsys,
        "run",
        "me.edits.split-chapters",
        "-v",
        "source=in.mkv",
        "-v",
        "dest=out.mkv",
    )
    assert code == 1 and "ffmpeg not found" in err


def test_a_recipe_name_is_not_looked_up_when_the_text_is_sql(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Text starting with one of the four words is SQL whatever else it might
    # have matched.
    _project(
        tmp_path,
        files={"src/own.sql": NORMALIZE, "queries/split.sql": RECIPE},
        manifest=_BIN,
    )
    code, out, _err = _run(tmp_path, monkeypatch, capsys, "compile", _MEDIA_QUERY)
    assert code == 0
    assert "scale=" in out


def test_a_leading_comment_does_not_hide_that_the_text_is_sql(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _project(
        tmp_path,
        files={"src/own.sql": NORMALIZE, "queries/split.sql": RECIPE},
        manifest=_BIN,
    )
    code, out, _err = _run(
        tmp_path, monkeypatch, capsys, "compile", f"-- a header\n/* and a block */\n{_MEDIA_QUERY}"
    )
    assert code == 0
    assert "scale=" in out


def test_a_bare_name_two_packages_ship_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    linked = tmp_path / "dev"
    _library(linked, "tracks", "0.5")
    (linked / "queries").mkdir()
    (linked / "queries" / "split.sql").write_text(RECIPE, encoding="utf-8")
    written = json.loads((linked / "ffrwd.json").read_text(encoding="utf-8"))
    written["bin"] = {"split-chapters": "queries/split.sql"}
    (linked / "ffrwd.json").write_text(json.dumps(written) + "\n", encoding="utf-8")

    project = tmp_path / "work"
    _project(
        project,
        files={"src/own.sql": NORMALIZE, "queries/split.sql": RECIPE},
        manifest=_BIN,
    )
    _lock(project, [_link(linked)])
    code, _out, err = _run(project, monkeypatch, capsys, "validate", "split-chapters")
    assert code == 1
    assert "more than one package ships a recipe named 'split-chapters'" in err
    assert "me.edits.split-chapters, tracks.lib.split-chapters" in err

    code, _out, err = _run(
        project,
        monkeypatch,
        capsys,
        "validate",
        "tracks.lib.split-chapters",
        "-v",
        "source=in.mkv",
        "-v",
        "dest=out.mkv",
    )
    assert code == 0


def test_an_undefined_variable_names_what_the_recipe_declares(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _project(
        tmp_path,
        files={"src/own.sql": NORMALIZE, "queries/split.sql": RECIPE},
        manifest=_BIN,
    )
    code, _out, err = _run(
        tmp_path, monkeypatch, capsys, "validate", "split-chapters", "-v", "source=in.mkv"
    )
    assert code == 1
    assert "':dest' was not set" in err
    assert "'split-chapters' declares source, dest" in err


def test_a_name_matching_nothing_fails_as_sql_and_names_the_recipes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _project(
        tmp_path,
        files={"src/own.sql": NORMALIZE, "queries/split.sql": RECIPE},
        manifest=_BIN,
    )
    code, _out, err = _run(tmp_path, monkeypatch, capsys, "compile", "split-chapter")
    assert code == 1
    assert "hint: installed recipes: me.edits.split-chapters" in err


def test_a_failing_query_is_not_offered_a_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _project(
        tmp_path,
        files={"src/own.sql": NORMALIZE, "queries/split.sql": RECIPE},
        manifest=_BIN,
    )
    code, _out, err = _run(tmp_path, monkeypatch, capsys, "compile", "SELECT nope(1)")
    assert code == 1
    assert "installed recipes" not in err


BAD_RECIPE = "SELECT nope(1)\n"


def test_a_recipe_that_resolved_is_not_offered_the_other_recipes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The rejection came from inside the recipe; listing the others is noise."""
    _project(
        tmp_path,
        files={"src/own.sql": NORMALIZE, "queries/split.sql": BAD_RECIPE},
        manifest=_BIN,
    )
    code, _out, err = _run(tmp_path, monkeypatch, capsys, "compile", "split-chapters")
    assert code == 1
    assert "installed recipes" not in err


def test_a_recipe_named_for_a_statement_word_is_refused(tmp_path: Path) -> None:
    _project(
        tmp_path,
        files={"src/own.sql": NORMALIZE, "queries/split.sql": RECIPE},
        manifest={"bin": {"select": "queries/split.sql"}},
    )
    error = _refuses(tmp_path, "recipe name 'select' is a word a query begins with")
    assert "rename it" in (error.hint or "")


# ---------------------------------------------------------------------------
# `ffrwd path`
# ---------------------------------------------------------------------------


def test_path_prints_the_store_directory_and_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A build script reads the line, so it is the path and no decoration."""
    entry = _installed(_library(tmp_path / "dev", "broadcast", "0.5", package="tracks"))
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [entry])

    code, out, err = _run(project, monkeypatch, capsys, "path", "broadcast/tracks")
    assert code == 0 and err == ""
    printed = Path(out.rstrip("\n"))
    assert out.count("\n") == 1
    assert printed == store.store_dir() / str(entry["store"])
    assert read_manifest(printed / "ffrwd.json").name == "broadcast/tracks"


def test_path_follows_a_link_to_the_directory_it_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _library(tmp_path / "dev", "broadcast", "0.5", package="tracks")
    project = tmp_path / "work"
    _project(project, files={"src/own.sql": NORMALIZE})
    _lock(project, [_link(source)])

    code, out, _err = _run(project, monkeypatch, capsys, "path", "broadcast/tracks")
    assert code == 0
    assert Path(out.rstrip("\n")) == source.resolve()


def test_path_refuses_a_package_the_lockfile_does_not_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _project(tmp_path, files={"src/own.sql": NORMALIZE})
    _lock(tmp_path, [])
    code, out, err = _run(tmp_path, monkeypatch, capsys, "path", "broadcast/tracks")
    assert code == 1 and out == ""
    assert "pins no package 'broadcast/tracks'" in err
    assert "ffrwd install broadcast/tracks" in err


def test_path_reads_the_machine_wide_lockfile_with_g(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    entry = _installed(_library(tmp_path / "dev", "broadcast", "0.5", package="tracks"))
    _lock(store.global_lock_path().parent, [entry])
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    code, out, _err = _run(elsewhere, monkeypatch, capsys, "path", "broadcast/tracks", "-g")
    assert code == 0
    assert Path(out.rstrip("\n")) == store.store_dir() / str(entry["store"])


def test_path_outside_a_project_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    code, _out, err = _run(elsewhere, monkeypatch, capsys, "path", "broadcast/tracks")
    assert code == 2
    assert "no ffrwd.lock" in err and "ffrwd path -g" in err


def test_path_refuses_something_that_is_not_a_package_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _project(tmp_path, files={"src/own.sql": NORMALIZE})
    _lock(tmp_path, [])
    code, _out, err = _run(tmp_path, monkeypatch, capsys, "path", "tracks")
    assert code == 1
    assert "is not a package name" in err


# ---------------------------------------------------------------------------
# `ffrwd publish`
# ---------------------------------------------------------------------------


def test_publish_outside_a_package_says_where_it_looked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The usage half: what publishing does with a package is test_publish.py's."""
    code, out, err = _run(tmp_path, monkeypatch, capsys, "publish")
    assert code == 2
    assert out == ""
    assert f"no {MANIFEST_NAME} in {tmp_path}" in err
    assert "ffrwd init" in err


# ---------------------------------------------------------------------------
# what a manifest carries for the registry: keywords, engines, models
# ---------------------------------------------------------------------------

_MODEL = {
    "repo": "depth-anything/small",
    "revision": "v1",
    "file": "onnx/model.onnx",
    "sha256": "a" * 64,
}


def test_the_registry_keys_are_read_and_carried(tmp_path: Path) -> None:
    """Read, validated and put on the package; no compile acts on any of them."""
    manifest = _project(
        tmp_path,
        files={"src/tracks.sql": QUIETER},
        manifest={
            "keywords": ["audio", "loudness"],
            "capabilities": ["nn", "http"],
            "ffrwd": ">=0.9",
            # An expression, not just an identifier: both are one line of text.
            "license": "MIT OR Apache-2.0",
            "homepage": "https://example.com/tracks",
            "models": {"quieter": dict(_MODEL)},
        },
    )
    package = read_manifest(manifest)
    assert package.keywords == ("audio", "loudness")
    assert package.capabilities == ("http", "nn")
    assert package.engines == ">=0.9"
    assert package.license == "MIT OR Apache-2.0"
    assert package.homepage == "https://example.com/tracks"
    assert dict(package.models) == {
        "quieter": (
            ModelPin(
                repo="depth-anything/small",
                revision="v1",
                file="onnx/model.onnx",
                sha256="a" * 64,
            ),
        )
    }


def test_a_model_of_several_files_is_read_in_the_order_it_is_written(
    tmp_path: Path,
) -> None:
    """The first pin is the graph; the rest are the files it refers to by name."""
    manifest = _project(
        tmp_path,
        files={"src/tracks.sql": QUIETER},
        manifest={
            "models": {
                "quieter": [
                    {**_MODEL, "file": "onnx/model.onnx"},
                    {**_MODEL, "file": "onnx/model.onnx.data", "sha256": "b" * 64},
                ]
            }
        },
    )
    pins = read_manifest(manifest).models["quieter"]
    assert [(pin.file, pin.filename, pin.sha256) for pin in pins] == [
        ("onnx/model.onnx", "model.onnx", "a" * 64),
        ("onnx/model.onnx.data", "model.onnx.data", "b" * 64),
    ]


def test_a_model_written_as_one_file_and_as_a_list_of_one_read_the_same(
    tmp_path: Path,
) -> None:
    """The two spellings are the same pin: a list is how a second file is added."""
    one = read_manifest(
        _project(
            tmp_path / "one",
            files={"src/tracks.sql": QUIETER},
            manifest={"models": {"quieter": dict(_MODEL)}},
        )
    )
    listed = read_manifest(
        _project(
            tmp_path / "listed",
            files={"src/tracks.sql": QUIETER},
            manifest={"models": {"quieter": [dict(_MODEL)]}},
        )
    )
    assert dict(one.models) == dict(listed.models)


def test_a_manifest_declaring_none_of_them_carries_none(tmp_path: Path) -> None:
    package = read_manifest(_project(tmp_path, files={"src/tracks.sql": QUIETER}))
    assert (
        package.keywords,
        package.license,
        package.homepage,
        package.capabilities,
        package.engines,
        dict(package.models),
    ) == ((), None, None, (), None, {})


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_homepage_accepts_http_and_https(tmp_path: Path, scheme: str) -> None:
    manifest = _project(
        tmp_path,
        files={"src/tracks.sql": QUIETER},
        manifest={"homepage": f"{scheme}://example.com/tracks"},
    )
    package = read_manifest(manifest)
    assert package.homepage == f"{scheme}://example.com/tracks"


def test_a_blank_homepage_is_the_same_as_absent(tmp_path: Path) -> None:
    manifest = _project(
        tmp_path, files={"src/tracks.sql": QUIETER}, manifest={"homepage": "   "}
    )
    assert read_manifest(manifest).homepage is None


@pytest.mark.parametrize(
    ("declared", "needle"),
    [
        ({"keywords": "audio"}, '"keywords" must be a list'),
        ({"keywords": [""]}, "keyword '' is not a label"),
        ({"keywords": ["a" * 33]}, "is longer than 32 characters"),
        ({"keywords": ["k"] * 17}, '"keywords" declares 17, and at most 16 are read'),
        ({"capabilities": "nn"}, '"capabilities" must be a list'),
        ({"capabilities": ["gpu"]}, "capability 'gpu' is not one this ffrwd grants"),
        ({"capabilities": [1]}, "capability 1 is not one this ffrwd grants"),
        ({"license": ""}, '"license" must be a non-empty string'),
        ({"license": "   "}, '"license" must be a non-empty string'),
        ({"license": ["MIT"]}, '"license" must be a non-empty string'),
        ({"license": "M" * 65}, '"license" is longer than 64 characters'),
        ({"license": "MIT\nApache-2.0"}, "is not one line of text"),
        ({"homepage": 1}, '"homepage" must be a string'),
        (
            {"homepage": "ftp://example.com/tracks"},
            '"homepage" \'ftp://example.com/tracks\' must start with http:// or https://',
        ),
        (
            {"homepage": "javascript:alert(1)"},
            '"homepage" \'javascript:alert(1)\' must start with http:// or https://',
        ),
        (
            # Exactly 301 characters: one past the cap.
            {"homepage": "https://" + "a" * 293},
            '"homepage" is longer than 300 characters',
        ),
        ({"ffrwd": ""}, '"ffrwd" must be a non-empty string'),
        ({"ffrwd": ">=0.9 or whatever you like"}, "is not a version range"),
        ({"ffrwd": ">" * 65}, '"ffrwd" is longer than 64 characters'),
        ({"models": {"Quieter": dict(_MODEL)}}, "model name 'Quieter' is not a plain identifier"),
        ({"models": {"quieter": "model.onnx"}}, "model 'quieter' must be a JSON object"),
        (
            {"models": {"quieter": {**_MODEL, "repo": "small"}}},
            "names the repository 'small', which is not <owner>/<name>",
        ),
        (
            {"models": {"quieter": {**_MODEL, "revision": "../main"}}},
            "names the revision '../main', which is not a branch, tag or commit",
        ),
        (
            {"models": {"quieter": {**_MODEL, "file": "../../etc/passwd"}}},
            "names the file '../../etc/passwd', which is not a file inside the revision",
        ),
        (
            {"models": {"quieter": {**_MODEL, "sha256": "nope"}}},
            "model 'quieter': 'nope' is not a sha256 digest",
        ),
        (
            {"models": {"quieter": {k: v for k, v in _MODEL.items() if k != "sha256"}}},
            'model \'quieter\' must name a non-empty "sha256"',
        ),
        (
            {"models": {"quieter": {**_MODEL, "url": "https://example/model"}}},
            "unknown key 'url' in model 'quieter'",
        ),
        ({"models": {"quieter": []}}, "model 'quieter' lists no file"),
        (
            {"models": {"quieter": [dict(_MODEL), {**_MODEL, "repo": "small"}]}},
            "model 'quieter' entry 2 names the repository 'small'",
        ),
        (
            {"models": {"quieter": [dict(_MODEL), {**_MODEL, "file": "onnx/"}]}},
            "model 'quieter' entry 2 names 'onnx/', which ends in no plain filename",
        ),
        (
            {"models": {"quieter": [dict(_MODEL), {**_MODEL, "file": "sub\\a.onnx"}]}},
            "model 'quieter' entry 2 names 'sub\\\\a.onnx', which ends in "
            "no plain filename",
        ),
        (
            {"models": {"quieter": [dict(_MODEL), {**_MODEL, "file": "a/."}]}},
            "model 'quieter' entry 2 names 'a/.', which ends in no plain filename",
        ),
        (
            {"models": {"quieter": [dict(_MODEL), {**_MODEL, "file": "b/quieter.onnx"}]}},
            "model 'quieter' entry 2 would land under 'quieter.onnx', where the first "
            "entry lands",
        ),
        (
            {
                "models": {
                    "quieter": [
                        dict(_MODEL),
                        {**_MODEL, "file": "a/weights.bin"},
                        {**_MODEL, "file": "b/weights.bin"},
                    ]
                }
            },
            "model 'quieter' entries 2 and 3 would both land under 'weights.bin'",
        ),
    ],
)
def test_a_registry_key_this_ffrwd_cannot_read_is_refused(
    tmp_path: Path, declared: dict[str, object], needle: str
) -> None:
    manifest = _project(tmp_path, files={"src/tracks.sql": QUIETER}, manifest=declared)
    with pytest.raises(FfrwdError) as caught:
        read_manifest(manifest)
    assert needle in caught.value.message, caught.value.message
