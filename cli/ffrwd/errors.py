"""Error types for ffrwd.

Every rejection produced by the library is a `FfrwdError` carrying a typed
`ErrorCode`, a human message, optional line/col anchoring, and an optional
hint. See the "Error contract" section of ffrwd-project.md.
"""

from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    PARSE_ERROR = "PARSE_ERROR"
    UNKNOWN_FUNCTION = "UNKNOWN_FUNCTION"
    UNKNOWN_ALIAS = "UNKNOWN_ALIAS"
    UDF_ARG_TYPE = "UDF_ARG_TYPE"
    SINGLE_OUTPUT_ONLY = "SINGLE_OUTPUT_ONLY"  # retired; never raised
    NO_STREAMING_EQUIVALENT = "NO_STREAMING_EQUIVALENT"
    CONCAT_MISMATCH = "CONCAT_MISMATCH"  # reserved; never raised
    UNSUPPORTED_SQL = "UNSUPPORTED_SQL"  # catch-all for constructs outside dialect
    STREAM_NOT_FOUND = "STREAM_NOT_FOUND"  # probed subscript out of range
    INPUT_NOT_FOUND = "INPUT_NOT_FOUND"  # */splat/broadcast with no readable input
    BROADCAST_MISMATCH = "BROADCAST_MISMATCH"  # zip length mismatch across arrays
    ROW_COUNT_MISMATCH = "ROW_COUNT_MISMATCH"  # several rows, one destination path
    UNKNOWN_SINK_OPTION = "UNKNOWN_SINK_OPTION"  # name not in sink.SINK_OPTIONS
    SINK_OPTION_TYPE = "SINK_OPTION_TYPE"  # value vs declared str/int/bool
    UNKNOWN_FILTER_OPTION = "UNKNOWN_FILTER_OPTION"  # `name => value` the filter lacks
    FILTER_OPTION_TYPE = "FILTER_OPTION_TYPE"  # value vs introspected type/range/consts
    UNKNOWN_INPUT_OPTION = "UNKNOWN_INPUT_OPTION"  # name not in inputs.INPUT_OPTIONS
    INPUT_OPTION_TYPE = "INPUT_OPTION_TYPE"  # value vs declared str/int/bool/num
    NOTHING_TO_SHOW = "NOTHING_TO_SHOW"  # --show/--show-only, no video-bearing output
    PLAYER_NOT_FOUND = "PLAYER_NOT_FOUND"  # --show asked for, ffplay not on PATH
    UNBOUNDED_LIVE_INPUT = "UNBOUNDED_LIVE_INPUT"  # one-open input, uncountable paths
    BUFFER_OVERFLOW = "BUFFER_OVERFLOW"  # a run-time edge outgrew its computed bound
    STARTUP_DEADLOCK = "STARTUP_DEADLOCK"  # no pipe order lets every process start
    INTERNAL = "INTERNAL"  # bug backstop; fuzz asserts this never fires


class FfrwdError(Exception):
    """Structured, machine-readable error for every user-input rejection.

    Library code that consumes user input must never raise anything other
    than this.
    """

    code: ErrorCode
    message: str
    line: int | None
    col: int | None
    hint: str | None

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        line: int | None = None,
        col: int | None = None,
        hint: str | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.line = line
        self.col = col
        self.hint = hint
        super().__init__(str(self))

    def to_dict(self) -> dict[str, object]:
        return {
            "line": self.line,
            "col": self.col,
            "code": self.code.value,
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
