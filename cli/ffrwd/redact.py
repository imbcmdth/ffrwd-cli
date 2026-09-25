"""Secrets kept out of what ffrwd prints.

A sidecar's ``-params`` can carry a relay's token, and a URL can carry one in
its path (``https://relay/<JWT>``), its query (``?jwt=...``) or its user
info. Every command line ffrwd prints or writes, and every process output it
hands on, passes through here first, with each secret spelled :data:`MASK`.
What is passed to a process is never touched: only what is shown.

A secret is recognised by where it sits or by its shape:

- the value of a JSON key named in :data:`SECRET_KEYS`, at any depth, or
  whose last word is one (``access_token``, ``apiKey``);
- the value of a URL query parameter named the same way, and a URL's
  password;
- a JWT anywhere: three base64url segments joined by dots whose first
  decodes to a JSON object, which is what a JOSE header is. Decoding it is
  what keeps a host name or a file name of three dotted words readable.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from collections.abc import Sequence

__all__ = ["MASK", "SECRET_KEYS", "argument", "argv", "is_secret_key", "text", "value"]

MASK = "***"

SECRET_KEYS = frozenset(
    {"token", "jwt", "password", "secret", "key", "auth", "authorization", "credentials"}
)

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WORDS = re.compile(r"[\s_.\-]+")

_JWT = re.compile(
    r"(?<![A-Za-z0-9_.\-])"
    r"(?P<header>[A-Za-z0-9_\-]{4,})\.[A-Za-z0-9_\-]{2,}\.[A-Za-z0-9_\-]*"
    r"(?![A-Za-z0-9_\-]|\.[A-Za-z0-9_\-])"
)
_QUERY = re.compile(r"(?P<lead>[?&;](?P<name>[A-Za-z0-9_.\-]+)=)(?P<value>[^&#;\s\"'<>]+)")
_USERINFO = re.compile(r"(?P<lead>[A-Za-z][A-Za-z0-9+.\-]*://[^:/@\s]+:)[^@/\s]+(?=@)")
_JSON_PAIR = re.compile(
    r"(?P<lead>\"(?P<name>[^\"\\]{1,64})\"\s*:\s*)"
    r"(?P<value>\"(?:[^\"\\]|\\.)*\"|-?\d[\d.eE+\-]*|true|false)"
)


def is_secret_key(name: str) -> bool:
    """Whether a value under `name` is a secret: the whole name, or its last
    word, is one of :data:`SECRET_KEYS`, whatever the case."""
    words = [word for word in _WORDS.split(_CAMEL.sub("_", name).lower()) if word]
    return name.lower() in SECRET_KEYS or (bool(words) and words[-1] in SECRET_KEYS)


def _is_jwt_header(segment: str) -> bool:
    padded = segment + "=" * (-len(segment) % 4)
    try:
        decoded = json.loads(base64.urlsafe_b64decode(padded))
    except (binascii.Error, ValueError):
        return False
    return isinstance(decoded, dict)


def _jwt(match: re.Match[str]) -> str:
    return MASK if _is_jwt_header(match["header"]) else match[0]


def _query(match: re.Match[str]) -> str:
    return match["lead"] + MASK if is_secret_key(match["name"]) else match[0]


def _json_pair(match: re.Match[str]) -> str:
    if not is_secret_key(match["name"]) or match["value"] == '""':
        return match[0]
    return f'{match["lead"]}"{MASK}"'


def text(written: str) -> str:
    """`written` with every secret in it masked: JWTs, secret query values
    and URL passwords, and secret values of JSON pairs spelled inline."""
    masked = _JWT.sub(_jwt, written)
    masked = _QUERY.sub(_query, masked)
    masked = _USERINFO.sub(lambda match: match["lead"] + MASK, masked)
    return _JSON_PAIR.sub(_json_pair, masked)


def value(parsed: object) -> object:
    """A parsed JSON value with every secret in it masked. A secret key's
    whole value goes, object or not; an empty one says nothing and stays."""
    if isinstance(parsed, dict):
        return {
            name: MASK if is_secret_key(name) and item not in ("", None) else value(item)
            for name, item in parsed.items()
        }
    if isinstance(parsed, list):
        return [value(item) for item in parsed]
    if isinstance(parsed, str):
        return text(parsed)
    return parsed


def argument(word: str) -> str:
    """One argv word as it may be shown. A JSON object or array, which is
    what ``-params`` carries, is masked by key and written back the way the
    sidecar renderer writes it; any other word as :func:`text` masks it."""
    if word.lstrip().startswith(("{", "[")):
        try:
            parsed = json.loads(word)
        except ValueError:
            return text(word)
        masked = value(parsed)
        return word if masked == parsed else json.dumps(masked)
    return text(word)


def argv(words: Sequence[str]) -> list[str]:
    """A command line as it may be shown, word for word."""
    return [argument(word) for word in words]
