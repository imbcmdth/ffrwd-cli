"""The token that authenticates this machine to the hosted registry.

``FFRWD_TOKEN`` wins when it is set. Otherwise the token is read out of a
credentials file under the user's own configuration directory:
``%APPDATA%\\ffrwd`` on Windows, ``$XDG_CONFIG_HOME`` or ``~/.config`` under
``ffrwd`` everywhere else. Deliberately not the cache directory the store
uses -- a token is not disposable, and clearing a cache must not log anyone
out.

The file is one JSON object, written with owner-only permissions and replaced
in one step::

    {"format_version": 1, "api": "https://...", "token": "ffrwd_..."}

Nothing here reaches the network, and nothing writes to stdout or stderr. A
file that is there and is not this shape is a typed rejection naming it: a
token the user believes they saved, silently ignored, is the one failure this
module refuses to have.
"""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from pathlib import Path

from .errors import ErrorCode, FfrwdError

__all__ = [
    "CREDENTIALS_NAME",
    "FORMAT_VERSION",
    "TOKEN_ENV",
    "TOKEN_HINT",
    "clear",
    "credentials_path",
    "is_token",
    "load",
    "save",
]

# The environment variable that overrides the file. What CI uses.
TOKEN_ENV = "FFRWD_TOKEN"

CREDENTIALS_NAME = "credentials.json"

# Bump on any change to what the file holds. Another version is refused rather
# than read as this one.
FORMAT_VERSION = 1

# A token is `ffrwd_` and 32 bytes base64url-encoded without padding.
_TOKEN_RE = re.compile(r"ffrwd_[A-Za-z0-9_-]{43}")

TOKEN_HINT = (
    "a token is `ffrwd_` followed by 43 base64url characters; mint one in the "
    "dashboard and pass it to `ffrwd login --token`"
)

_LOGIN_HINT = "run `ffrwd login --token <token>` to write it again"


def _reject(message: str, hint: str) -> FfrwdError:
    return FfrwdError(ErrorCode.UNSUPPORTED_SQL, message, hint=hint)


def is_token(text: str) -> bool:
    """True when `text` is spelled like an ffrwd token. Shape only."""
    return _TOKEN_RE.fullmatch(text) is not None


def _windows_config_dir() -> Path:
    """``%APPDATA%\\ffrwd``, or where APPDATA itself normally points."""
    appdata = os.environ.get("APPDATA")
    if appdata and appdata.strip():
        return Path(appdata) / "ffrwd"
    return Path.home() / "AppData" / "Roaming" / "ffrwd"


def _posix_config_dir() -> Path:
    """``$XDG_CONFIG_HOME/ffrwd``, or ``~/.config/ffrwd``."""
    written = os.environ.get("XDG_CONFIG_HOME")
    base = Path(written) if written and written.strip() else Path.home() / ".config"
    return base / "ffrwd"


def _config_dir() -> Path:
    """The directory the credentials file sits in. The one home-directory seam."""
    return _windows_config_dir() if os.name == "nt" else _posix_config_dir()


def credentials_path() -> Path:
    """Where the token is written, whether or not anything is there yet."""
    return _config_dir() / CREDENTIALS_NAME


def load() -> str | None:
    """The token to send, or None when this machine has none.

    The environment wins over the file, so a shell can hand one command a
    different identity without touching what is stored.
    """
    written = os.environ.get(TOKEN_ENV)
    if written and written.strip():
        return written.strip()
    path = credentials_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:  # nothing saved, which is not a failure
        return None
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError) as err:
        raise _reject(f"{path} is not valid JSON", _LOGIN_HINT) from err
    if not isinstance(data, dict):
        raise _reject(f"{path} is not a JSON object", _LOGIN_HINT)
    if data.get("format_version") != FORMAT_VERSION:
        raise _reject(
            f"{path} was written in credentials format "
            f"{data.get('format_version')!r}, and this ffrwd reads {FORMAT_VERSION}",
            _LOGIN_HINT,
        )
    token = data.get("token")
    if not isinstance(token, str) or not is_token(token):
        raise _reject(f"{path} holds no ffrwd token", _LOGIN_HINT)
    return token


def save(token: str, *, api: str) -> Path:
    """Write `token` for `api`, owner-readable only, replacing whatever was there.

    Written beside the destination and moved onto it, so a reader sees the old
    file or the new one. The permissions are set on the temporary file before
    anything is written into it, so the token is never briefly world-readable.
    """
    if not is_token(token):
        raise _reject(f"{token!r} is not an ffrwd token", TOKEN_HINT)
    path = credentials_path()
    payload = json.dumps(
        {"format_version": FORMAT_VERSION, "api": api, "token": token}, indent=2
    ) + "\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f"{CREDENTIALS_NAME}-", suffix=".tmp"
        )
    except OSError as err:
        raise _reject(
            f"{path} could not be written: {err.strerror or err}",
            "check that the directory exists and is writable",
        ) from err
    try:
        os.chmod(temporary, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as file:
            file.write(payload)
        os.replace(temporary, path)
    except OSError as err:
        try:
            os.unlink(temporary)
        except OSError:  # already gone, or the directory refuses us twice
            pass
        raise _reject(
            f"{path} could not be written: {err.strerror or err}",
            "check that the directory exists and is writable",
        ) from err
    return path


def clear() -> bool:
    """Remove the credentials file. True when one was there to remove."""
    path = credentials_path()
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError as err:
        raise _reject(
            f"{path} could not be removed: {err.strerror or err}",
            "remove the file by hand to log out",
        ) from err
    return True
