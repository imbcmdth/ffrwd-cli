"""Warnings: what a compile has to say without refusing to compile.

Every rejection ffrwd produces is a `FfrwdError`, and that left nothing to
say about a compile that succeeded but cost something. Two package
resolutions cost something worth naming: a call that landed on a package
installed machine-wide rather than in the project the query lives in, and a
call into a LINKED directory, whose files are edited in place and so are not
pinned by any lockfile.

The channel is an optional ``on_warning`` callback threaded beside
``packages`` through the compile entry points, the same shape
:func:`ffrwd.execute.execute` already uses for ``echo``: the CLI passes a
sink it prints to stderr, the MCP tools pass one they return as a
``warnings`` array, tests pass a list's ``append``, and the default is None --
silence. No library code prints.

A compile warns at most once per package. A CLI command or an MCP tool
compiles the same text more than once (the table-query fallback), so
:class:`WarningLog` drops the repeats a second compile produces.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

__all__ = ["OnWarning", "FfrwdWarning", "WarningCode", "WarningLog"]


class WarningCode(str, Enum):
    GLOBAL_PACKAGE = "GLOBAL_PACKAGE"  # resolved outside the project the query is in
    LINKED_PACKAGE = "LINKED_PACKAGE"  # a linked directory, so no lockfile pins it
    EMPTY_STREAM_ARRAY = "EMPTY_STREAM_ARRAY"  # a column the file has no tracks for
    IGNORE_PATTERN = "IGNORE_PATTERN"  # a .gitignore line the packer does not read
    MISSING_README = "MISSING_README"  # a package published with no README.md
    MISSING_LICENSE = "MISSING_LICENSE"  # a package published with no "license"
    RECIPE_DOES_NOT_COMPILE = "RECIPE_DOES_NOT_COMPILE"  # a recipe the probe compile rejected


@dataclass(frozen=True)
class FfrwdWarning:
    """One diagnostic that is not a rejection.

    Shaped like `FfrwdError` on purpose -- a typed code, a human message, an
    optional anchor and hint -- so an editor renders both the same way.
    `package` is the name of the package the warning is about, and what a
    sink dedupes on.
    """

    code: WarningCode
    package: str
    message: str
    line: int | None = None
    col: int | None = None
    hint: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "line": self.line,
            "col": self.col,
            "code": self.code.value,
            "package": self.package,
            "message": self.message,
            "hint": self.hint,
        }

    def __str__(self) -> str:
        parts: list[str] = []
        if self.line is not None and self.col is not None:
            parts.append(f"line {self.line}:{self.col}: ")
        elif self.line is not None:
            parts.append(f"line {self.line}: ")
        parts.append(f"{self.code.value}: {self.message}")
        if self.hint is not None:
            parts.append(f" (hint: {self.hint})")
        return "".join(parts)


OnWarning = Callable[[FfrwdWarning], None]


class WarningLog:
    """A sink that keeps the first warning of each code per package, in order."""

    def __init__(self) -> None:
        self._seen: set[tuple[WarningCode, str]] = set()
        self._warnings: list[FfrwdWarning] = []

    def __call__(self, warning: FfrwdWarning) -> None:
        key = (warning.code, warning.package)
        if key in self._seen:
            return
        self._seen.add(key)
        self._warnings.append(warning)

    @property
    def warnings(self) -> tuple[FfrwdWarning, ...]:
        return tuple(self._warnings)
