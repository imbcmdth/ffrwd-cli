"""Tests for the token this machine publishes with, and for `login`/`logout`.

Nothing here reaches the network -- `login` deliberately does not either. The
suite-wide isolation in ``conftest`` points the credentials file at a
directory of this test's own and clears ``FFRWD_TOKEN``, so no developer's
real token is ever read and no check leaves the next one logged in.

The two tests that pin WHERE the file goes need the real answer rather than
the redirected one, so they call the function this module captured at import,
before the fixture replaced it.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from ffrwd import cli, credentials
from ffrwd.errors import FfrwdError

TOKEN = "ffrwd_" + "b" * 43

# Captured before the suite-wide fixture redirects it.
_REAL_CONFIG_DIR = credentials._config_dir


def _run(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *argv: str,
) -> tuple[int, str, str]:
    monkeypatch.chdir(root)
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ---------------------------------------------------------------------------
# what a token looks like
# ---------------------------------------------------------------------------


def test_a_token_is_the_prefix_and_thirty_two_bytes_of_base64url() -> None:
    written = {
        "ffrwd_" + "A" * 43: True,
        "ffrwd_" + "-_9aZ" + "A" * 38: True,  # every base64url character is one
        "ffrwd_" + "A" * 42: False,  # 31 bytes is not 32
        "ffrwd_" + "A" * 44: False,
        "ffrwd_" + "A" * 42 + "+": False,  # standard base64, not url-safe
        "A" * 43: False,  # no prefix
        "ffrwd-" + "A" * 43: False,
        "": False,
    }
    assert {text: credentials.is_token(text) for text in written} == written


# ---------------------------------------------------------------------------
# where the file goes
# ---------------------------------------------------------------------------


def test_on_windows_the_file_sits_under_the_roaming_application_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    assert credentials._windows_config_dir() == tmp_path / "Roaming" / "ffrwd"


def test_elsewhere_the_file_sits_under_the_configuration_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Not under the cache: a cleared cache must not log anyone out."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    assert credentials._posix_config_dir() == tmp_path / "config" / "ffrwd"

    monkeypatch.delenv("XDG_CONFIG_HOME")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    assert credentials._posix_config_dir() == tmp_path / "home" / ".config" / "ffrwd"


def test_the_file_this_machine_reads_is_the_one_its_own_family_names() -> None:
    expected = (
        credentials._windows_config_dir()
        if os.name == "nt"
        else credentials._posix_config_dir()
    )
    assert _REAL_CONFIG_DIR() == expected


# ---------------------------------------------------------------------------
# saving, reading and removing
# ---------------------------------------------------------------------------


def test_nothing_saved_and_nothing_in_the_environment_is_no_token() -> None:
    assert credentials.load() is None


def test_a_saved_token_reads_back_out_of_a_file_only_its_owner_may_read() -> None:
    path = credentials.save(TOKEN, api="https://api.example")
    assert path == credentials.credentials_path()
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "format_version": credentials.FORMAT_VERSION,
        "api": "https://api.example",
        "token": TOKEN,
    }
    assert credentials.load() == TOKEN
    if os.name != "nt":  # Windows has no mode bits to check
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_the_environment_wins_over_the_saved_file(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials.save(TOKEN, api="https://api.example")
    monkeypatch.setenv(credentials.TOKEN_ENV, "ffrwd_" + "c" * 43)
    assert credentials.load() == "ffrwd_" + "c" * 43


def test_saving_refuses_something_that_is_not_a_token() -> None:
    with pytest.raises(FfrwdError) as caught:
        credentials.save("hunter2", api="https://api.example")
    assert "is not an ffrwd token" in caught.value.message
    assert not credentials.credentials_path().exists()


def test_clearing_removes_the_file_and_says_when_there_was_none() -> None:
    credentials.save(TOKEN, api="https://api.example")
    assert credentials.clear() is True
    assert not credentials.credentials_path().exists()
    assert credentials.clear() is False


@pytest.mark.parametrize(
    ("written", "needle"),
    [
        ("{not json", "is not valid JSON"),
        ("[]", "is not a JSON object"),
        ('{"format_version": 99, "token": "x"}', "was written in credentials format 99"),
        ('{"format_version": 1, "token": "hunter2"}', "holds no ffrwd token"),
    ],
)
def test_a_credentials_file_this_ffrwd_cannot_read_is_refused(
    written: str, needle: str
) -> None:
    """A token the user believes they saved is never silently ignored."""
    path = credentials.credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(written, encoding="utf-8")
    with pytest.raises(FfrwdError) as caught:
        credentials.load()
    assert needle in caught.value.message
    assert "ffrwd login" in (caught.value.hint or "")


# ---------------------------------------------------------------------------
# `ffrwd login` / `ffrwd logout`
# ---------------------------------------------------------------------------


def test_login_saves_the_token_and_reaches_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = _run(tmp_path, monkeypatch, capsys, "login", "--token", TOKEN)
    assert code == 0
    assert err == ""
    assert str(credentials.credentials_path()) in out
    assert credentials.load() == TOKEN


def test_login_refuses_a_token_of_the_wrong_shape_as_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = _run(tmp_path, monkeypatch, capsys, "login", "--token", "hunter2")
    assert code == 2
    assert out == ""
    assert "that is not an ffrwd token" in err
    assert "43 base64url characters" in err
    assert credentials.load() is None


def test_logout_removes_the_token_and_says_so_when_there_was_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(tmp_path, monkeypatch, capsys, "login", "--token", TOKEN)[0] == 0
    code, out, _err = _run(tmp_path, monkeypatch, capsys, "logout")
    assert code == 0
    assert "removed the token" in out
    assert credentials.load() is None

    code, out, _err = _run(tmp_path, monkeypatch, capsys, "logout")
    assert code == 0
    assert "no token was saved" in out


def test_logout_says_the_environment_still_answers_for_this_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(credentials.TOKEN_ENV, TOKEN)
    code, _out, err = _run(tmp_path, monkeypatch, capsys, "logout")
    assert code == 0
    assert f"{credentials.TOKEN_ENV} is set in this environment" in err
    assert credentials.load() == TOKEN
