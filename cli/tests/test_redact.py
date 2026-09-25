"""Secrets kept out of what ffrwd prints: the masking itself, and every place
a command line or a process's output is shown or written.

Unit tier: nothing spawned. Every token here is made up.
"""

from __future__ import annotations

import base64
import json
import shlex
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from ffrwd import cli, redact
from ffrwd.execute import PlanResult, ProcessResult, StageResult, plan_argv, render_plan
from ffrwd.ir import Graph, Node, Output, SinkUnit
from ffrwd.processes import SidecarProcess, external_ids, partition


def _segment(obj: object) -> str:
    raw = json.dumps(obj, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


JWT = f"{_segment({'alg': 'HS256', 'typ': 'JWT'})}.{_segment({'sub': 'live'})}.c2lnbmF0dXJl"
MASK = redact.MASK


def _params(**fields: object) -> str:
    """A ``-params`` word as the sidecar renderer writes one."""
    return json.dumps(fields, sort_keys=True)


def test_every_kind_of_secret_is_masked_and_nothing_else() -> None:
    table: list[tuple[str, str]] = [
        # -params: a secret key's value, whatever its case or depth.
        (
            _params(relay="moqt://203.0.113.7:4443", token="abc123"),
            _params(relay="moqt://203.0.113.7:4443", token=MASK),
        ),
        (_params(Token="abc123", JWT="x"), _params(JWT=MASK, Token=MASK)),
        (
            _params(auth={"user": "u", "password": "p"}, hops=[{"secret": "s", "n": 1}]),
            _params(auth=MASK, hops=[{"n": 1, "secret": MASK}]),
        ),
        (
            _params(access_token="a", apiKey="b", Authorization="c", credentials=["d"]),
            _params(Authorization=MASK, access_token=MASK, apiKey=MASK, credentials=MASK),
        ),
        # An empty token says nothing, and a key that only contains a
        # secret word is not one.
        (
            _params(token="", keyframes=2, audio_group_ms=200),
            _params(token="", keyframes=2, audio_group_ms=200),
        ),
        # A JWT: bare, in a URL's path or query, and inside -params.
        (JWT, MASK),
        (f"https://relay.example/{JWT}", f"https://relay.example/{MASK}"),
        (f"https://relay.example/live?jwt={JWT}&x=1", f"https://relay.example/live?jwt={MASK}&x=1"),
        (
            _params(broadcast="live/a", relay=f"https://relay.example/{JWT}"),
            _params(broadcast="live/a", relay=f"https://relay.example/{MASK}"),
        ),
        # A secret query value that is no JWT, and a URL's password.
        (
            "https://cdn.example/a.m3u8?token=opaque&w=1",
            f"https://cdn.example/a.m3u8?token={MASK}&w=1",
        ),
        ("rtmp://user:hunter2@ingest.example/app", f"rtmp://user:{MASK}@ingest.example/app"),
        # Three dotted words that are not a JWT, and a four-part host.
        ("relay.example.net", "relay.example.net"),
        ("film.720p.mp4", "film.720p.mp4"),
        (
            "https://draft-16.cloudflare.mediaoverquic.com",
            "https://draft-16.cloudflare.mediaoverquic.com",
        ),
        ("203.0.113.7", "203.0.113.7"),
        ("scale=w=1280:h=720", "scale=w=1280:h=720"),
    ]
    assert [(word, redact.argument(word)) for word, _ in table] == table


def test_what_a_process_wrote_is_masked_where_it_says_a_secret() -> None:
    stderr = (
        f"relay 'https://relay.example/{JWT}': refused\n"
        'params {"token": "abc123", "rows": "summary"}\n'
        "GET /live.m3u8?token=opaque HTTP/1.1\n"
    )
    assert redact.text(stderr) == (
        f"relay 'https://relay.example/{MASK}': refused\n"
        f'params {{"token": "{MASK}", "rows": "summary"}}\n'
        f"GET /live.m3u8?token={MASK} HTTP/1.1\n"
    )


def _argv() -> list[str]:
    return [
        "ffrwd-wasm",
        "-m",
        "publish.wasm",
        "-params",
        _params(relay=f"https://r.example/{JWT}", token="abc123"),
    ]


def test_a_member_is_reported_masked_and_run_as_it_was_given() -> None:
    argv = _argv()
    member = ProcessResult(
        id="sidecar3", argv=argv, exit_code=1, stderr=f"publish: relay https://r.example/{JWT}"
    )
    assert member.argv == _argv(), "what runs is never touched"
    reported = cli._member_error(member)
    assert reported.startswith("error: sidecar3 exited with code 1\n  ffrwd-wasm")
    assert "abc123" not in reported and JWT not in reported
    assert f'"token": "{MASK}"' in reported
    assert member.stderr_tail == f"publish: relay https://r.example/{MASK}"


def test_a_compile_listing_masks_what_the_run_passes() -> None:
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["e0"] = Node(id="e0", filter="negate", args={}, inputs=["src:a:v:0"], outputs=["video"])
    g.sinks = [
        SinkUnit(outputs=[Output(ref="e0", type="video", name=None, metadata={})], path="out.mp4")
    ]
    plan = partition(g, external=external_ids("e0"))

    def stand_in(
        process: SidecarProcess, reads: Sequence[str] = (), writes: Sequence[str] = ()
    ) -> list[str]:
        return [*_argv(), "-f", "nut", "-i", *(reads or ("pipe:0",)), "-f", "nut", "pipe:1"]

    listed = render_plan(plan, sidecar_argv=stand_in)
    assert "abc123" not in listed and JWT not in listed
    assert shlex.quote(_params(relay=f"https://r.example/{MASK}", token=MASK)) in listed
    run = plan_argv(plan, sidecar_argv=stand_in)
    assert _params(relay=f"https://r.example/{JWT}", token="abc123") in run["sidecar0"]


def test_a_plain_compile_masks_its_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    command = ["ffmpeg", "-i", f"https://cdn.example/a.m3u8?jwt={JWT}", "out.mp4"]
    monkeypatch.setattr(cli, "build_ffmpeg_commands", lambda emitted: [list(command)])
    emitted = SimpleNamespace(measure_filter_complex=None)
    assert cli._shell_commands([emitted]) == [  # type: ignore[list-item]
        shlex.join(["ffmpeg", "-i", f"https://cdn.example/a.m3u8?jwt={MASK}", "out.mp4"])
    ]


def test_verbose_echoes_are_masked(capsys: pytest.CaptureFixture[str]) -> None:
    cli._echo_member("sidecar0", _argv())
    cli._echo_command(["ffmpeg", "-i", f"https://relay.example/{JWT}"])
    echoed = capsys.readouterr().out
    assert "abc123" not in echoed and JWT not in echoed
    assert echoed.count(MASK) == 3


def test_a_dump_file_is_masked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FFRWD_DUMP_STDERR", str(tmp_path))
    member = ProcessResult(
        id="sidecar2", argv=_argv(), exit_code=1, stderr=f"subscribe: https://r.example/{JWT}\n"
    )
    cli._debug_dump_stderr(PlanResult(stages=[StageResult(index=0, members=[member])]))
    assert (tmp_path / "sidecar2.stderr").read_text(encoding="utf-8") == (
        f"exit=1 terminated=False\nsubscribe: https://r.example/{MASK}\n"
    )
