"""Tests for ffrwd/loudnorm.py -- the two-pass loudnorm handoff.

One parser serves both consumers, so it is tested once here: the
``loudnorm2env`` subcommand (tests/test_cli.py) and ``run``'s in-process
substitution both call :func:`ffrwd.loudnorm.parse` on the same text.

``FFMPEG_STDERR`` below is a real capture -- ffmpeg 9.0.1 measuring
tests/fixtures/av.mp4 -- trimmed only of the stream-listing lines in the
middle, which is exactly the shape the parser has to cope with: log output
first, the JSON block last, and a ``[Parsed_loudnorm_0 @ ...]`` prefix line
between them.
"""

from __future__ import annotations

import pytest

from ffrwd import loudnorm

FFMPEG_STDERR = """\
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'tests/fixtures/av.mp4':
  Duration: 00:00:02.00, start: 0.000000, bitrate: 290 kb/s
Stream mapping:
  Stream #0:1 (aac) -> loudnorm:default
  loudnorm:default -> Stream #0:0 (pcm_s16le)
Press [q] to stop, [?] for help
Output #0, null, to 'pipe:':
  Stream #0:0: Audio: pcm_s16le, 192000 Hz, mono, s16, 3072 kb/s
[Parsed_loudnorm_0 @ 000001d04f81d540]\x20
{
\t"input_i" : "-21.76",
\t"input_tp" : "-17.69",
\t"input_lra" : "0.00",
\t"input_thresh" : "-31.76",
\t"output_i" : "-16.05",
\t"output_tp" : "-11.93",
\t"output_lra" : "0.00",
\t"output_thresh" : "-26.05",
\t"normalization_type" : "linear",
\t"target_offset" : "0.05"
}
[out#0/null @ 000001d04f80aa00] video:0KiB audio:750KiB subtitle:0KiB
size=N/A time=00:00:02.00 bitrate=N/A speed= 166x elapsed=0:00:00.01
"""

MEASURED = {
    "FFRWD_LN_I": "-21.76",
    "FFRWD_LN_TP": "-17.69",
    "FFRWD_LN_LRA": "0.00",
    "FFRWD_LN_THRESH": "-31.76",
    "FFRWD_LN_OFFSET": "0.05",
}


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------


def test_parse_reads_real_ffmpeg_stderr() -> None:
    assert loudnorm.parse(FFMPEG_STDERR) == MEASURED


def test_parse_takes_the_input_measurements_not_the_predicted_output() -> None:
    """``input_*`` is what pass 2 needs; ``output_*`` is loudnorm's own
    prediction of what pass 2 will achieve, and feeding it back would
    normalize the file twice."""
    assert loudnorm.parse(FFMPEG_STDERR)["FFRWD_LN_I"] == "-21.76"


def test_parse_takes_the_last_block_when_two_runs_are_concatenated() -> None:
    second = FFMPEG_STDERR.replace("-21.76", "-30.00")
    assert loudnorm.parse(FFMPEG_STDERR + second)["FFRWD_LN_I"] == "-30.00"


def test_parse_survives_a_json_object_in_the_log_noise() -> None:
    noise = '[info] config: {"threads": 4}\n'
    assert loudnorm.parse(noise + FFMPEG_STDERR) == MEASURED


def test_parse_rejects_input_with_no_json_at_all() -> None:
    with pytest.raises(ValueError, match="no loudnorm JSON block"):
        loudnorm.parse("ffmpeg version 9.0.1\nConversion failed!\n")


def test_parse_rejects_input_with_no_json_when_empty() -> None:
    with pytest.raises(ValueError, match="no loudnorm JSON block"):
        loudnorm.parse("")


def test_parse_names_the_missing_keys() -> None:
    partial = FFMPEG_STDERR.replace('"input_thresh"', '"nope"')
    with pytest.raises(ValueError, match="missing input_thresh"):
        loudnorm.parse(partial)


def test_parse_rejects_a_non_numeric_value() -> None:
    broken = FFMPEG_STDERR.replace('"-21.76"', "null")
    with pytest.raises(ValueError, match="not a number"):
        loudnorm.parse(broken)


def test_parse_accepts_json_numbers_too() -> None:
    numeric = FFMPEG_STDERR.replace('"-21.76"', "-21.76")
    assert loudnorm.parse(numeric)["FFRWD_LN_I"] == "-21.76"


# ---------------------------------------------------------------------------
# the two phases' arguments
# ---------------------------------------------------------------------------


def test_measure_phase_adds_only_print_format() -> None:
    written = {"I": -16, "TP": -1.5, "LRA": 11}
    assert loudnorm.phase_args(written, measure=True) == {
        "I": -16,
        "TP": -1.5,
        "LRA": 11,
        "print_format": "json",
    }


def test_correct_phase_adds_the_five_measurements_and_linear() -> None:
    assert list(loudnorm.phase_args({"I": -16}, measure=False)) == [
        "I",
        "measured_I",
        "measured_TP",
        "measured_LRA",
        "measured_thresh",
        "offset",
        "linear",
    ]


def test_correct_phase_values_are_shell_variable_references() -> None:
    args = loudnorm.phase_args({}, measure=False)
    assert args["measured_I"] == "${FFRWD_LN_I}"
    assert args["offset"] == "${FFRWD_LN_OFFSET}"
    assert args["linear"] == "true"


def test_phase_args_renders_only_what_was_written() -> None:
    """Every option is optional: an omitted one keeps loudnorm's own default
    rather than being written out with a guessed value."""
    assert loudnorm.phase_args({}, measure=True) == {"print_format": "json"}


def test_phase_args_does_not_mutate_the_node_arguments() -> None:
    written: dict[str, object] = {"I": -16}
    loudnorm.phase_args(written, measure=False)
    assert written == {"I": -16}


# ---------------------------------------------------------------------------
# the shell handoff: adjacent-quote splicing
# ---------------------------------------------------------------------------


def test_shell_word_splices_a_reference_out_of_the_quotes() -> None:
    word = "[0:a:0]loudnorm=measured_I=${FFRWD_LN_I}:linear=true[out0]"
    assert loudnorm.shell_word(word) == (
        "'[0:a:0]loudnorm=measured_I='\"${FFRWD_LN_I}\"':linear=true[out0]'"
    )


def test_shell_word_leaves_an_ordinary_word_to_shlex() -> None:
    assert loudnorm.shell_word("-map") == "-map"
    assert loudnorm.shell_word("[out0]") == "'[out0]'"


def test_shell_word_splices_every_reference() -> None:
    word = ":".join(f"{m.arg}=${{{m.var}}}" for m in loudnorm.MEASURED)
    spliced = loudnorm.shell_word(word)
    for entry in loudnorm.MEASURED:
        assert f"'\"${{{entry.var}}}\"'" in spliced


def test_shell_word_survives_a_quote_in_the_same_word() -> None:
    """shlex.quote breaks a `'` out into `'\"'\"'`, so the reference still
    lands inside a single-quoted run and the splice stays well-formed."""
    word = "drawtext=text=it\\'s:x=${FFRWD_LN_I}"
    spliced = loudnorm.shell_word(word)
    assert "'\"${FFRWD_LN_I}\"'" in spliced


def test_shell_join_is_shlex_join_for_a_command_with_no_references() -> None:
    argv = ["ffmpeg", "-i", "in.mp4", "-map", "[out0]", "out.mp4"]
    assert loudnorm.shell_join(argv) == "ffmpeg -i in.mp4 -map '[out0]' out.mp4"


def test_measure_command_pipes_stderr_into_the_subcommand() -> None:
    assert loudnorm.measure_command("ffmpeg -f null -") == (
        'eval "$(ffmpeg -f null - 2>&1 | ffrwd loudnorm2env)"'
    )


# ---------------------------------------------------------------------------
# export lines and in-process substitution: the two ends of the same handoff
# ---------------------------------------------------------------------------


def test_export_lines_names_every_variable_in_order() -> None:
    assert loudnorm.export_lines(MEASURED).splitlines() == [
        "export FFRWD_LN_I=-21.76",
        "export FFRWD_LN_TP=-17.69",
        "export FFRWD_LN_LRA=0.00",
        "export FFRWD_LN_THRESH=-31.76",
        "export FFRWD_LN_OFFSET=0.05",
    ]


def test_export_lines_quotes_a_value_that_needs_it() -> None:
    values = dict(MEASURED, FFRWD_LN_I="a b")
    assert "export FFRWD_LN_I='a b'" in loudnorm.export_lines(values)


def test_substitute_replaces_references_in_place() -> None:
    word = "loudnorm=measured_I=${FFRWD_LN_I}:offset=${FFRWD_LN_OFFSET}"
    assert loudnorm.substitute(word, MEASURED) == (
        "loudnorm=measured_I=-21.76:offset=0.05"
    )


def test_substitute_leaves_a_word_with_no_reference_alone() -> None:
    assert loudnorm.substitute("-map", MEASURED) == "-map"


def test_substitute_with_no_measurements_is_the_identity() -> None:
    word = "loudnorm=measured_I=${FFRWD_LN_I}"
    assert loudnorm.substitute(word, {}) == word


def test_parse_round_trips_through_substitute() -> None:
    """The two consumers agree by construction: what the parser produces is
    keyed by exactly the variables the correction phase references."""
    values = loudnorm.parse(FFMPEG_STDERR)
    args = loudnorm.phase_args({}, measure=False)
    for entry in loudnorm.MEASURED:
        rendered = args[entry.arg]
        assert isinstance(rendered, str)
        assert loudnorm.substitute(rendered, values) == values[entry.var]
