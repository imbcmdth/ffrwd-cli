"""What a run holds of each member's stderr (ffrwd.execute._Log).

Unit tier: the streams are in-memory pipes and plain bytes.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

_EXECUTE = sys.modules["ffrwd.execute"]


def test_a_log_holds_the_end_of_what_was_written() -> None:
    """Whole chunks go from the front once the rest holds `keep` bytes, so
    the end is always there and the log never grows past one chunk more."""
    log = _EXECUTE._Log(keep=100)
    for index in range(1000):
        log.append(b"%04d " % index + b"x" * 45 + b"\n")
    text = log.text()
    assert text.endswith("0999 " + "x" * 45 + "\n")
    assert 100 <= len(text) <= 100 + 51


def test_a_log_short_of_its_bound_holds_everything() -> None:
    log = _EXECUTE._Log(keep=1 << 18)
    log.append(b"Input #0\n")
    log.append(b"Conversion failed!\n")
    assert log.text() == "Input #0\nConversion failed!\n"


def _write_past_the_bound(log: object) -> bytes:
    """800 KiB of numbered lines into `log`, more than it holds in memory,
    with a password in the first line; what was written comes back."""
    lines = [b"opening https://user:hunter2@example.com/in.mp4\n"]
    lines += [b"frame=%8d fps= 30\r" % n for n in range(800 * 1024 // 20)]
    for line in lines:
        log.append(line)  # type: ignore[attr-defined]
    return b"".join(lines)


def test_a_dumped_run_writes_its_whole_log_to_a_file_and_holds_its_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FFRWD_DUMP_STDERR asks for every member's whole log. It goes to a file
    as it arrives, so the run still holds only the end of it in memory."""
    monkeypatch.setenv("FFRWD_DUMP_STDERR", str(tmp_path))
    log = _EXECUTE._Log.for_run()
    written = _write_past_the_bound(log)
    log.close()
    assert log.whole is not None and log.whole.parent == tmp_path
    assert log.whole.read_bytes() == written
    held = log.text().encode()
    assert written.endswith(held)
    assert _EXECUTE._STDERR_KEEP <= len(held) <= _EXECUTE._STDERR_KEEP + 20


def test_the_dump_is_the_whole_log_masked_and_its_file_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FFRWD_DUMP_STDERR", str(tmp_path))
    log = _EXECUTE._Log.for_run()
    written = _write_past_the_bound(log)
    log.close()
    result = _EXECUTE.ProcessResult(
        id="ffmpeg0", argv=[], exit_code=1, stderr=log.text(), stderr_whole=log.whole
    )
    dumped = tmp_path / "ffmpeg0.stderr"
    _EXECUTE.write_stderr_dump(dumped, result)
    text = dumped.read_text(encoding="utf-8")
    assert text.startswith("exit=1 terminated=False\n")
    assert "hunter2" not in text
    assert text.count("frame=") == written.count(b"frame=")
    assert list(tmp_path.glob("*.part")) == []


def test_a_log_with_no_file_dumps_what_it_held(tmp_path: Path) -> None:
    result = _EXECUTE.ProcessResult(
        id="sidecar0", argv=[], exit_code=0, stderr="opening\nthe stream ended\n"
    )
    dumped = tmp_path / "sidecar0.stderr"
    _EXECUTE.write_stderr_dump(dumped, result)
    assert dumped.read_text(encoding="utf-8") == (
        "exit=0 terminated=False\nopening\nthe stream ended\n"
    )


def test_an_undumped_run_writes_no_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FFRWD_DUMP_STDERR", raising=False)
    log = _EXECUTE._Log.for_run()
    _write_past_the_bound(log)
    assert log.whole is None
    assert len(log.text()) <= _EXECUTE._STDERR_KEEP + 64


def test_a_drain_keeps_a_copy_of_each_read_at_its_own_size() -> None:
    """Each kept chunk is its own object of the size read, never one buffer
    shared between reads."""
    written = b"".join(b"line %d\n" % n for n in range(50_000))
    stream = io.BufferedReader(io.BytesIO(written), buffer_size=1 << 16)
    into: list[bytes] = []
    _EXECUTE._drain(stream, into)
    assert b"".join(into) == written
    assert all(type(chunk) is bytes for chunk in into)
    assert len({id(chunk) for chunk in into}) == len(into)
