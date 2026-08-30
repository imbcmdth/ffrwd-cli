"""Tests for docs/errors.md and docs/error-schema.json.

(a) every ErrorCode value appears as a heading in docs/errors.md.
(b) docs/error-schema.json validates the dicts FfrwdError.to_dict()
    actually produces. jsonschema is not a project dependency, so the
    check is hand-rolled rather than using a full validator library.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ffrwd.errors import ErrorCode, FfrwdError

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS_DIR = _REPO_ROOT / "docs"


# ---------------------------------------------------------------------------
# (a) every ErrorCode appears as a heading in docs/errors.md
# ---------------------------------------------------------------------------


def _errors_md_text() -> str:
    return (_DOCS_DIR / "errors.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("code", list(ErrorCode))
def test_every_error_code_documented(code: ErrorCode) -> None:
    text = _errors_md_text()
    assert f"## {code.value}" in text


def test_errors_md_has_no_stray_code_headings() -> None:
    # Every "## " heading that looks like a SCREAMING_SNAKE_CASE code should
    # be a real ErrorCode -- catches typos or stale renames.
    text = _errors_md_text()
    known = {code.value for code in ErrorCode}
    for line in text.splitlines():
        if line.startswith("## ") and line[3:].isupper():
            assert line[3:] in known, f"undocumented/unknown code heading: {line!r}"


def test_errors_md_uses_lf_newlines_only() -> None:
    data = (_DOCS_DIR / "errors.md").read_bytes()
    assert b"\r" not in data


# ---------------------------------------------------------------------------
# (b) error-schema.json validates FfrwdError.to_dict() output
# ---------------------------------------------------------------------------

_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "null": type(None),
    "object": dict,
    "array": list,
    "boolean": bool,
}


def _load_schema() -> dict[str, object]:
    with open(_DOCS_DIR / "error-schema.json", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    return data


def _check_type(value: object, type_spec: object) -> bool:
    names = type_spec if isinstance(type_spec, list) else [type_spec]
    for name in names:
        assert isinstance(name, str)
        py_type = _JSON_TYPE_MAP[name]
        if name == "integer" and isinstance(value, bool):
            continue  # bool is a subclass of int but not a JSON "integer"
        if isinstance(value, py_type):
            return True
    return False


def _validate(instance: dict[str, object], schema: dict[str, object]) -> None:
    """Minimal, hand-rolled JSON Schema check.

    Covers only what error-schema.json actually uses: a flat object with
    per-property type/enum, a required list, and additionalProperties.
    Good enough to catch drift between the schema and to_dict() without
    taking a dependency on the jsonschema package.
    """
    assert schema["type"] == "object"
    properties = schema["properties"]
    assert isinstance(properties, dict)
    required = schema["required"]
    assert isinstance(required, list)

    for key in required:
        assert key in instance, f"missing required key {key!r}"

    if schema.get("additionalProperties") is False:
        for key in instance:
            assert key in properties, f"unexpected key {key!r}"

    for key, value in instance.items():
        prop_schema = properties[key]
        assert isinstance(prop_schema, dict)
        type_spec = prop_schema["type"]
        assert _check_type(value, type_spec), (
            f"key {key!r}: value {value!r} does not match type {type_spec!r}"
        )
        if "enum" in prop_schema:
            assert value in prop_schema["enum"], f"key {key!r}: {value!r} not in enum"


def test_schema_is_draft_2020_12() -> None:
    schema = _load_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def test_schema_requires_all_five_keys() -> None:
    schema = _load_schema()
    assert schema["required"] == ["line", "col", "code", "message", "hint"]


def test_schema_enum_matches_error_code() -> None:
    schema = _load_schema()
    properties = schema["properties"]
    assert isinstance(properties, dict)
    code_schema = properties["code"]
    assert isinstance(code_schema, dict)
    assert set(code_schema["enum"]) == {code.value for code in ErrorCode}


@pytest.mark.parametrize(
    ("code", "message", "line", "col", "hint"),
    [
        (ErrorCode.PARSE_ERROR, "Expecting )", 1, 33, None),
        (ErrorCode.UNKNOWN_FUNCTION, "unknown function sharpen()", 1, 8, "known functions: blur"),
        (ErrorCode.UNKNOWN_ALIAS, "unknown alias 'b'", 1, 8, "known names: a"),
        (
            ErrorCode.UDF_ARG_TYPE,
            "blur() expects blur(frame, num), got blur(frame)",
            1,
            8,
            "arguments are frame expressions or literals, in the order shown",
        ),
        (
            ErrorCode.SINGLE_OUTPUT_ONLY,
            "SELECT must produce exactly one frame column, got 2",
            1,
            17,
            None,
        ),
        (
            ErrorCode.NO_STREAMING_EQUIVALENT,
            "GROUP BY has no streaming equivalent",
            3,
            10,
            "remove the GROUP BY clause",
        ),
        (
            ErrorCode.CONCAT_MISMATCH,
            "fps/resolution mismatch across UNION ALL branches",
            None,
            None,
            None,
        ),
        (
            ErrorCode.UNSUPPORTED_SQL,
            "SELECT * is not supported",
            1,
            8,
            "select a single frame expression",
        ),
        (
            ErrorCode.INTERNAL,
            "internal error while lowering (ValueError: x)",
            1,
            1,
            "please report this query as a bug",
        ),
        # No line/col/hint at all -- the "empty query" case.
        (ErrorCode.PARSE_ERROR, "empty query", None, None, None),
    ],
)
def test_error_schema_validates_to_dict_output(
    code: ErrorCode,
    message: str,
    line: int | None,
    col: int | None,
    hint: str | None,
) -> None:
    schema = _load_schema()
    err = FfrwdError(code, message, line=line, col=col, hint=hint)
    _validate(err.to_dict(), schema)


def test_error_schema_rejects_missing_key() -> None:
    schema = _load_schema()
    err = FfrwdError(ErrorCode.PARSE_ERROR, "x", line=1, col=1)
    instance = err.to_dict()
    del instance["hint"]
    with pytest.raises(AssertionError):
        _validate(instance, schema)


def test_error_schema_rejects_unknown_code() -> None:
    schema = _load_schema()
    instance: dict[str, object] = {
        "line": 1,
        "col": 1,
        "code": "NOT_A_REAL_CODE",
        "message": "x",
        "hint": None,
    }
    with pytest.raises(AssertionError):
        _validate(instance, schema)


def test_error_schema_rejects_wrong_type() -> None:
    schema = _load_schema()
    instance: dict[str, object] = {
        "line": "1",  # should be an int, not a string
        "col": 1,
        "code": "PARSE_ERROR",
        "message": "x",
        "hint": None,
    }
    with pytest.raises(AssertionError):
        _validate(instance, schema)


# ---------------------------------------------------------------------------
# (c) ffrwd-json.schema.json agrees with the manifest reader
# ---------------------------------------------------------------------------


def _manifest_schema() -> dict[str, object]:
    with open(_DOCS_DIR / "ffrwd-json.schema.json", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    return data


def test_manifest_schema_names_exactly_the_known_keys() -> None:
    from ffrwd import project

    properties = _manifest_schema()["properties"]
    assert isinstance(properties, dict)
    assert set(properties) == set(project._KNOWN)


def test_manifest_schema_requires_what_the_reader_requires() -> None:
    from ffrwd import project

    required = _manifest_schema()["required"]
    assert isinstance(required, list)
    assert set(required) == set(project._REQUIRED)


def test_manifest_schema_rejects_unknown_keys() -> None:
    # The reader rejects a key outside _KNOWN, so the schema must too.
    assert _manifest_schema()["additionalProperties"] is False


def test_manifest_schema_capabilities_agree() -> None:
    from ffrwd import project

    properties = _manifest_schema()["properties"]
    assert isinstance(properties, dict)
    capabilities = properties["capabilities"]
    assert isinstance(capabilities, dict)
    items = capabilities["items"]
    assert isinstance(items, dict)
    assert set(items["enum"]) == set(project.CAPABILITIES)
