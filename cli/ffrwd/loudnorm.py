"""Two-pass loudnorm: the measured values, their names, and the JSON parser.

``ffrwd.loudnorm2(stream, I => ..., TP => ..., LRA => ...)`` compiles to a
SEQUENCE of two ffmpeg commands. Pass 1 runs the same stream through
``loudnorm=<written opts>:print_format=json``, muxes nothing (``-f null -``)
and prints its measurements as a JSON block on stderr. Pass 2 runs the real
encode with those five numbers fed back in as ``measured_*``/``offset``, plus
``linear=true``.

Two consumers carry the numbers from pass 1 to pass 2, and they share the
parser below:

* the printed command line is a POSIX-shell chain --
  ``eval "$(<pass 1> 2>&1 | ffrwd loudnorm2env)" && <pass 2>`` -- where
  ``loudnorm2env`` turns the JSON into ``export FFRWD_LN_*=`` lines and
  pass 2's filtergraph splices those variables in through adjacent-quote
  concatenation (:func:`shell_word`);
* ``run`` uses no shell at all: it captures pass 1's stderr, parses it here,
  and substitutes the values straight into pass 2's argv
  (:func:`substitute`).

Two costs come with that shape, and the docs carry both: the printed command
needs ``ffrwd loudnorm2env`` on PATH at run time (it is the one compiled
output that is not pure ffmpeg), and it is POSIX-shell only -- cmd.exe and
PowerShell users run the query with ``run``.
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass

__all__ = [
    "ENV_SUBCOMMAND",
    "FFMPEG_FILTER",
    "FILTER",
    "MEASURED",
    "OPTIONS",
    "export_lines",
    "measure_command",
    "parse",
    "phase_args",
    "shell_join",
    "shell_word",
    "substitute",
]

# The IR node's filter name. Deliberately NOT "loudnorm": it is a two-phase
# pseudo-filter, and emit is what turns it into a real `loudnorm` per phase.
# A bare `loudnorm(...)` call still compiles to an ordinary one-pass node.
FILTER = "loudnorm2"
FFMPEG_FILTER = "loudnorm"

# The macro's own named-only options, in render order.
OPTIONS = ("I", "TP", "LRA")

ENV_SUBCOMMAND = "loudnorm2env"

_PRINT_FORMAT = ("print_format", "json")
_LINEAR = ("linear", "true")


@dataclass(frozen=True)
class Measured:
    """One measurement carried from pass 1 to pass 2."""

    arg: str  # pass 2's loudnorm option name
    var: str  # the environment variable the printed command routes it through
    key: str  # loudnorm's own JSON key


MEASURED: tuple[Measured, ...] = (
    Measured("measured_I", "FFRWD_LN_I", "input_i"),
    Measured("measured_TP", "FFRWD_LN_TP", "input_tp"),
    Measured("measured_LRA", "FFRWD_LN_LRA", "input_lra"),
    Measured("measured_thresh", "FFRWD_LN_THRESH", "input_thresh"),
    Measured("offset", "FFRWD_LN_OFFSET", "target_offset"),
)

_REFERENCE_RE = re.compile(r"\$\{(" + "|".join(m.var for m in MEASURED) + r")\}")


def _reference(var: str) -> str:
    return "${" + var + "}"


def phase_args(written: Mapping[str, object], *, measure: bool) -> dict[str, object]:
    """The real ``loudnorm`` arguments for one phase.

    `written` is what the query actually wrote (nothing is defaulted in --
    an option the user omitted keeps loudnorm's own default). Pass 1 adds
    ``print_format=json``; pass 2 adds the five measurements as
    ``${FFRWD_LN_*}`` references plus ``linear=true``.
    """
    args: dict[str, object] = dict(written)
    if measure:
        args[_PRINT_FORMAT[0]] = _PRINT_FORMAT[1]
        return args
    for entry in MEASURED:
        args[entry.arg] = _reference(entry.var)
    args[_LINEAR[0]] = _LINEAR[1]
    return args


def parse(text: str) -> dict[str, str]:
    """loudnorm's measurements out of `text`, keyed by environment variable.

    `text` is ffmpeg's whole stderr: log lines first, the JSON block last.
    The LAST object carrying every key wins. Raises ``ValueError`` with a
    plain message when there is no such block.
    """
    found: dict[str, object] | None = None
    missing: list[str] = []
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _end = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        absent = [entry.key for entry in MEASURED if entry.key not in value]
        if absent:
            missing = absent
            continue
        found = value
    if found is None:
        if missing:
            raise ValueError(
                "loudnorm JSON is missing " + ", ".join(missing)
            )
        raise ValueError("no loudnorm JSON block in the input")
    return {entry.var: _scalar(found[entry.key]) for entry in MEASURED}


def _scalar(value: object) -> str:
    """One JSON value as text. loudnorm writes strings; numbers are tolerated."""
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise ValueError(f"loudnorm JSON value is not a number: {value!r}")
    return str(value)


def export_lines(values: Mapping[str, str]) -> str:
    """`values` as the shell ``export`` block ``loudnorm2env`` prints."""
    return "\n".join(
        f"export {entry.var}={shlex.quote(values[entry.var])}" for entry in MEASURED
    )


def substitute(word: str, values: Mapping[str, str]) -> str:
    """One argv word with its ``${FFRWD_LN_*}`` references replaced.

    ``run``'s half of the handoff: no shell is involved, so the expansion a
    shell would do happens here. A word with no reference comes back
    unchanged, which is every word of every other command.
    """
    if not values:
        return word
    return _REFERENCE_RE.sub(lambda m: values.get(m.group(1), m.group(0)), word)


def shell_word(word: str) -> str:
    """One argv word, POSIX-quoted, with ``${FFRWD_LN_*}`` left expandable.

    ``shlex.quote`` always single-quotes a word holding ``${...}``, which
    would stop the shell expanding it. Each reference is therefore spliced
    back out of the quotes as ``'"${VAR}"'`` -- close, expand double-quoted,
    reopen -- and the shell's adjacent-quote concatenation glues the word
    back into one argument.
    """
    return _REFERENCE_RE.sub(lambda m: "'\"" + m.group(0) + "\"'", shlex.quote(word))


def shell_join(argv: list[str]) -> str:
    """`argv` as one shell command line (``shlex.join`` plus the splice)."""
    return " ".join(shell_word(word) for word in argv)


def measure_command(command: str) -> str:
    """Pass 1 wrapped in the ``eval`` that exports what it measured."""
    return f'eval "$({command} 2>&1 | ffrwd {ENV_SUBCOMMAND})"'
