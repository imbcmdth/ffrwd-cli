"""Tests for the top-level ``ffrwd`` package: version and public API.

``ffrwd.__version__`` and ``ffrwd.__all__`` are the library's front page;
these tests check the front page matches the real package metadata and the
real source-module objects, not a hand-copied restatement.
"""

from __future__ import annotations

import importlib
import importlib.metadata

import ffrwd


def test_version_matches_installed_metadata() -> None:
    assert ffrwd.__version__ == importlib.metadata.version("ffrwd")


def test_all_is_sorted_and_complete() -> None:
    expected = {
        "compile_sql",
        "compile_commands",
        "compile_table_sql",
        "classify",
        "FfrwdError",
        "ErrorCode",
        "FfrwdWarning",
        "WarningCode",
        "emit",
        "build_ffmpeg_commands",
        "build_system_prompt",
        "execute",
        "ExecutionResult",
        "CommandResult",
        "probe",
        "discover",
        "PackageSet",
        "load_registry",
        "Registry",
        "render_table",
        "render_csv",
        "TableSink",
    }
    assert set(ffrwd.__all__) == expected
    assert ffrwd.__all__ == sorted(ffrwd.__all__)


def test_exports_are_the_source_module_objects() -> None:
    compiler = importlib.import_module("ffrwd.compiler")
    errors = importlib.import_module("ffrwd.errors")
    emit_module = importlib.import_module("ffrwd.emit")
    execute_module = importlib.import_module("ffrwd.execute")
    probe_module = importlib.import_module("ffrwd.probe")
    prompt_module = importlib.import_module("ffrwd.prompt")
    project_module = importlib.import_module("ffrwd.project")
    registry_module = importlib.import_module("ffrwd.registry")
    table_module = importlib.import_module("ffrwd.table")
    warnings_module = importlib.import_module("ffrwd.warnings")

    assert ffrwd.compile_sql is compiler.compile_sql
    assert ffrwd.compile_commands is compiler.compile_commands
    assert ffrwd.compile_table_sql is compiler.compile_table_sql
    assert ffrwd.classify is compiler.classify
    assert ffrwd.FfrwdError is errors.FfrwdError
    assert ffrwd.ErrorCode is errors.ErrorCode
    assert ffrwd.FfrwdWarning is warnings_module.FfrwdWarning
    assert ffrwd.WarningCode is warnings_module.WarningCode
    assert ffrwd.emit is emit_module.emit
    assert ffrwd.build_ffmpeg_commands is emit_module.build_ffmpeg_commands
    assert ffrwd.execute is execute_module.execute
    assert ffrwd.ExecutionResult is execute_module.ExecutionResult
    assert ffrwd.CommandResult is execute_module.CommandResult
    assert ffrwd.probe is probe_module.probe
    assert ffrwd.discover is project_module.discover
    assert ffrwd.PackageSet is project_module.PackageSet
    assert ffrwd.build_system_prompt is prompt_module.build_system_prompt
    assert ffrwd.load_registry is registry_module.load
    assert ffrwd.Registry is registry_module.Registry
    assert ffrwd.render_table is table_module.render_table
    assert ffrwd.render_csv is table_module.render_csv
    assert ffrwd.TableSink is table_module.TableSink
