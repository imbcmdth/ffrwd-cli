"""Tests for ``ffrwd run --remote`` and ``ffrwd jobs``.

Everything runs against a fake HTTP seam (``packages._urlopen``, the same
table-of-answers shape ``test_publish`` uses): no network, no ffmpeg, no
fixtures. The parser's syntactic helpers and ``probe.is_url`` are covered
here too, since the remote client is what they exist for.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import urllib.error
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest

import ffrwd
from ffrwd import cli, credentials, packages, parser, remote, store
from ffrwd.errors import FfrwdError
from ffrwd.probe import ProbeResult, is_url
from ffrwd.project import LinkEntry, RegistryEntry, links_path, write_linksfile, write_lockfile
from ffrwd.table import TableResult, render_table

API = "https://api.example"
JOBS_URL = f"{API}/functions/v1/jobs"
UPLOAD_URL = "https://runner.example/upload"
START_URL = "https://runner.example/start"
TOKEN = "ffrwd_" + "d" * 43
JOB_ID = "0a1b2c3d-1111-2222-3333-444455556666"
JOB_TOKEN = "payload.signature"

MEDIA_QUERY = "COPY (SELECT a.video[1] FROM input('in.mp4') a) TO 'out.mp4'"

REMAINING = {
    "period": "2026-08-01",
    "resets_on": "2026-09-01",
    "cpu_seconds": 2820,
    "gpu_seconds": 900,
    "cpu_cents": 188,
    "gpu_cents": 150,
    "cents": 338,
}


class _Fake:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = io.BytesIO(body)

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def __enter__(self) -> _Fake:
        return self

    def __exit__(self, *_exception: object) -> None:
        return None


class _Served:
    """The HTTP seam: a table of URL to answer, and what was asked for."""

    def __init__(self) -> None:
        self.answers: dict[str, object] = {}
        self.probes: dict[str, ProbeResult] = {}
        self.asked: list[tuple[str, dict[str, str], bytes | None]] = []

    def __call__(self, request: object, timeout: float | None = None) -> _Fake:
        url = str(request.full_url)  # type: ignore[attr-defined]
        headers = {
            str(key).lower(): str(value)
            for key, value in dict(request.headers).items()  # type: ignore[attr-defined]
        }
        self.asked.append((url, headers, request.data))  # type: ignore[attr-defined]
        answer = self.answers.get(url, (404, b'{"error": "not found"}'))
        status, content = answer if isinstance(answer, tuple) else (200, answer)
        assert isinstance(content, bytes)
        if status >= 400:
            raise urllib.error.HTTPError(url, status, "refused", {}, io.BytesIO(content))  # type: ignore[arg-type]
        return _Fake(status, content)

    def sent_to(self, url: str) -> tuple[dict[str, str], bytes | None]:
        for asked, headers, body in self.asked:
            if asked == url:
                return headers, body
        raise AssertionError(f"nothing was sent to {url}")


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Served]:
    fake = _Served()
    monkeypatch.setenv(packages.API_ENV, API)
    monkeypatch.setattr(packages, "_urlopen", fake)
    # ffprobe never runs in the unit tier: an input probes as whatever the
    # test put in `probes`, and as nothing at all otherwise.
    monkeypatch.setattr(remote, "probe", lambda path, *args, **kwargs: fake.probes.get(path))
    yield fake


@pytest.fixture
def logged_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(credentials.TOKEN_ENV, TOKEN)


def _submit_accepted(served: _Served, *, remaining: dict[str, object] | None = REMAINING) -> None:
    answer: dict[str, object] = {
        "job_id": JOB_ID,
        "job_token": JOB_TOKEN,
        "upload_url": UPLOAD_URL,
        "start_url": START_URL,
        "outputs_expire_days": 7,
    }
    if remaining is not None:
        answer["remaining"] = remaining
    served.answers[JOBS_URL] = json.dumps(answer).encode("utf-8")
    served.answers[START_URL] = (202, json.dumps({"state": "queued"}).encode("utf-8"))


def _run_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {"show": False, "show_only": False, "timeout": None}
    values.update(overrides)
    return argparse.Namespace(**values)


def _query(text: str, **overrides: object) -> cli._Query:
    values: dict[str, object] = {"text": text, "unset": {}, "recipe": None, "source": text}
    values.update(overrides)
    return cli._Query(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "probed",
    [ProbeResult(streams=[], duration=None), ProbeResult(streams=[], duration=30.0, live=True)],
    ids=["no-duration", "live"],
)
def test_an_input_ffprobe_calls_unbounded_is_refused_before_any_request(
    served: _Served,
    logged_in: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    probed: ProbeResult,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "in.mp4").write_bytes(b"x")
    served.probes["in.mp4"] = probed
    with pytest.raises(FfrwdError) as caught:
        remote.submit_run(_query(MEDIA_QUERY), None, _run_args())
    assert caught.value.message == (
        "input 'in.mp4' has no bounded duration, and the hosted runner takes only "
        "sources it can bound"
    )
    assert caught.value.hint == "drop --remote and run it locally"
    assert caught.value.line == 1
    assert served.asked == []


def test_a_bounded_input_submits_and_an_unprobeable_one_is_the_runners_call(
    served: _Served, logged_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "in.mp4").write_bytes(b"x")
    digest = hashlib.sha256(b"x").hexdigest()
    _submit_accepted(served)
    served.answers[f"{UPLOAD_URL}?sha256={digest}"] = json.dumps(
        {"already": False, "bytes": 1}
    ).encode("utf-8")
    served.probes["in.mp4"] = ProbeResult(streams=[], duration=12.5)
    remote.submit_run(_query(MEDIA_QUERY), None, _run_args())
    assert [asked for asked, _headers, _body in served.asked][0] == JOBS_URL
    served.asked.clear()
    # A URL this machine cannot probe passes through: the runner probes it.
    remote.submit_run(_query(STREAM_QUERY), None, _run_args())
    assert [asked for asked, _headers, _body in served.asked][0] == JOBS_URL


# ---------------------------------------------------------------------------
# the syntactic parser helpers
# ---------------------------------------------------------------------------


def test_input_specs_enumerates_every_literal_with_positions() -> None:
    sql = "SELECT a.video[1] FROM input('x.mp4') a, input('x.mp4') b"
    assert parser.input_specs(sql) == [
        parser.InputSpec(path="x.mp4", line=1, col=30),
        parser.InputSpec(path="x.mp4", line=1, col=48),
    ]


def test_input_specs_reads_scripts_and_urls() -> None:
    sql = (
        "CREATE VIEW v AS SELECT a.video[1] FROM input('in.mp4') a;\n"
        "COPY (SELECT b.audio[1] FROM input('rtmp://live/key', format => 'flv') b)"
        " TO 'out.mp4';\n"
    )
    assert parser.input_specs(sql) == [
        parser.InputSpec(path="in.mp4", line=1, col=47),
        parser.InputSpec(path="rtmp://live/key", line=2, col=36),
    ]


def test_input_specs_skips_non_literals_and_the_source_namespace() -> None:
    # A NULL path and `ffmpeg.input(...)` are other shapes' rejections to
    # make; neither is a file this walk may claim.
    assert parser.input_specs("SELECT a.video[1] FROM input(NULL) a") == []
    assert parser.input_specs("SELECT 1 FROM ffmpeg.input('x.mp4') t") == []


def test_copy_destinations_lists_literal_targets_in_order() -> None:
    sql = "COPY (SELECT 1) TO 'a.mp4'; COPY (SELECT 2) TO 'b.mkv'"
    assert parser.copy_destinations(sql) == ["a.mp4", "b.mkv"]


def test_copy_destinations_skips_stdout_and_computed_targets() -> None:
    sql = (
        "COPY (SELECT 1) TO STDOUT WITH (FORMAT csv);\n"
        "COPY (SELECT c.index FROM input('f.mkv') f, unnest(f.chapters) c)"
        " TO (c.index::text || '.mkv');\n"
        "COPY (SELECT 2) TO 'out.mp4';\n"
    )
    assert parser.copy_destinations(sql) == ["out.mp4"]


def test_is_url_truth_table() -> None:
    for spec in ("rtmp://live/key", "udp://0.0.0.0:9999", "https://cdn.example/x.mp4"):
        assert is_url(spec)
    for spec in ("out.mp4", "C:/media/x.mp4", "dir/sub/x.mp4", ""):
        assert not is_url(spec)


def test_check_output_dir_still_reads_urls_through_the_helper(tmp_path: Path) -> None:
    assert cli._check_output_dir("udp://host:9999") is None
    missing = tmp_path / "nowhere" / "out.mp4"
    said = cli._check_output_dir(str(missing))
    assert said is not None and str(missing.parent) in said


# ---------------------------------------------------------------------------
# the refusals, all before any request
# ---------------------------------------------------------------------------


def test_a_remote_run_without_a_token_is_refused(
    served: _Served, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["run", "--remote", MEDIA_QUERY])
    err = capsys.readouterr().err
    assert code == 1
    assert "running remotely needs an ffrwd token, and this machine holds none" in err
    assert "mint a token with the run scope" in err
    assert served.asked == []


def test_a_table_query_is_refused_before_anything_remote(
    served: _Served, logged_in: None, capsys: pytest.CaptureFixture[str]
) -> None:
    table_query = "SELECT t.codec FROM input('x.mkv') f, unnest(f.audio) t"
    code = cli.main(["run", "--remote", table_query])
    err = capsys.readouterr().err
    assert code == 1
    assert "a table query needs no cloud -- run it locally" in err
    assert served.asked == []


@pytest.mark.parametrize("flag", ["--show", "--show-only"])
def test_show_and_remote_conflict(
    served: _Served, logged_in: None, capsys: pytest.CaptureFixture[str], flag: str
) -> None:
    code = cli.main(["run", "--remote", flag, MEDIA_QUERY])
    err = capsys.readouterr().err
    assert code == 1
    assert "--show opens a window on this machine, and a remote run has none" in err
    assert served.asked == []


def test_a_computed_destination_submits(
    served: _Served, logged_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _submit_accepted(served)
    text = (
        "COPY (SELECT c.index FROM input('rtmp://live/key') f, unnest(f.chapters) c)"
        " TO (c.index::text || '.mkv')"
    )
    remote.submit_run(_query(text), None, _run_args())
    _headers, body = served.sent_to(JOBS_URL)
    assert body is not None
    payload = json.loads(body)
    # The runner discovers the real paths from the compiled graph; the
    # syntactic view declares none for a computed destination.
    assert payload["outputs"] == []
    assert payload["format_version"] == 2


def test_a_missing_file_input_is_refused_with_its_anchor(
    served: _Served, logged_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FfrwdError) as caught:
        remote.submit_run(_query(MEDIA_QUERY), None, _run_args())
    assert caught.value.message == "input 'in.mp4' does not exist"
    assert (caught.value.line, caught.value.col) == (1, 36)
    assert served.asked == []


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


def test_submit_posts_the_spec_uploads_the_file_and_starts(
    served: _Served,
    logged_in: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "in.mp4").write_bytes(b"media bytes")
    digest = hashlib.sha256(b"media bytes").hexdigest()
    _submit_accepted(served)
    served.answers[f"{UPLOAD_URL}?sha256={digest}"] = json.dumps(
        {"already": False, "bytes": 11}
    ).encode("utf-8")

    query = (
        "COPY (SELECT a.video[1] FROM input('in.mp4') a, input('in.mp4') again, "
        "input('https://cdn.example/x.mp4') b, input('rtmp://live/key') c) TO 'out.mp4'"
    )
    code = cli.main(["run", "--remote", "--timeout", "120", query])
    captured = capsys.readouterr()
    assert code == 0
    # The free line sits between the job id and the follow line, on stdout
    # like the rest of the result.
    assert captured.out == (
        f"submitted {JOB_ID}\n"
        "free this month: 47m CPU + 15m GPU remaining\n"
        f"follow: ffrwd jobs --watch    fetch: ffrwd jobs --fetch {JOB_ID[:8]}\n"
    )
    # Each step narrates on stderr, sizes and all; stdout stays the result.
    assert "submitting the job\n" in captured.err
    assert "uploading in.mp4 (11 bytes)\n" in captured.err
    assert "starting the job\n" in captured.err

    headers, body = served.sent_to(JOBS_URL)
    assert headers["authorization"] == f"Bearer {TOKEN}"
    assert body is not None
    assert json.loads(body) == {
        "format_version": 2,
        "query": query,
        "variables": {},
        "recipe": None,
        "owner": None,
        "lock": None,
        "packages": [],
        "inputs": [
            {"path": "in.mp4", "kind": "file", "sha256": digest, "bytes": 11},
            {"path": "https://cdn.example/x.mp4", "kind": "url"},
            {"path": "rtmp://live/key", "kind": "url"},
        ],
        "outputs": ["out.mp4"],
        "timeout_s": 120.0,
        "client_version": ffrwd.__version__,
    }
    upload_headers, upload_body = served.sent_to(f"{UPLOAD_URL}?sha256={digest}")
    assert upload_headers["x-job-token"] == JOB_TOKEN
    assert upload_headers["content-type"] == "application/octet-stream"
    assert upload_body == b"media bytes"
    start_headers, _start_body = served.sent_to(START_URL)
    assert start_headers["x-job-token"] == JOB_TOKEN


def test_a_recipe_run_carries_its_owner_and_the_lock_text(
    served: _Served,
    logged_in: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "recipes").mkdir()
    (tmp_path / "recipes" / "duck.sql").write_text(
        "COPY (SELECT f.audio[1] FROM input(:'source') f) TO :'dest';\n",
        encoding="utf-8",
    )
    (tmp_path / "ffrwd.json").write_text(
        json.dumps(
            {
                "name": "broadcast/tracks",
                "version": "1.0.0",
                "bin": {"duck": "recipes/duck.sql"},
            }
        ),
        encoding="utf-8",
    )
    write_lockfile(tmp_path / "ffrwd.lock", ())
    lock_text = (tmp_path / "ffrwd.lock").read_text(encoding="utf-8")
    (tmp_path / "in.mkv").write_bytes(b"tracks")
    digest = hashlib.sha256(b"tracks").hexdigest()
    _submit_accepted(served)
    served.answers[f"{UPLOAD_URL}?sha256={digest}"] = json.dumps(
        {"already": False, "bytes": 6}
    ).encode("utf-8")

    code = cli.main(
        ["run", "--remote", "duck", "-v", "source=in.mkv", "-v", "dest=out.mkv"]
    )
    assert code == 0, capsys.readouterr().err
    _headers, body = served.sent_to(JOBS_URL)
    assert body is not None
    payload = json.loads(body)
    assert payload["recipe"] == "duck"
    assert payload["owner"] == ["broadcast/tracks", "1.0.0"]
    assert payload["lock"] == lock_text
    assert payload["outputs"] == ["out.mkv"]
    assert payload["inputs"] == [
        {"path": "in.mkv", "kind": "file", "sha256": digest, "bytes": 6}
    ]
    # The -v pairs travel raw beside the query they were substituted into.
    assert payload["variables"] == {"source": "in.mkv", "dest": "out.mkv"}
    assert "in.mkv" in payload["query"] and ":'source'" not in payload["query"]


def test_variables_travel_raw_beside_the_substituted_query(
    served: _Served, logged_in: None
) -> None:
    text = "COPY (SELECT a.video[1] FROM input('rtmp://live/key') a) TO 'out.mp4'"
    _submit_accepted(served)
    remote.submit_run(
        _query(text, variables={"source": "rtmp://live/key"}), None, _run_args()
    )
    _headers, body = served.sent_to(JOBS_URL)
    assert body is not None
    payload = json.loads(body)
    assert payload["variables"] == {"source": "rtmp://live/key"}
    assert payload["query"] == text


# ---------------------------------------------------------------------------
# the effective lock: project entries over the machine-wide lockfile's
# ---------------------------------------------------------------------------

STREAM_QUERY = "COPY (SELECT a.video[1] FROM input('rtmp://live/key') a) TO 'out.mp4'"


def _registry_entry(name: str, version: str, sha256: str) -> RegistryEntry:
    return RegistryEntry(
        name=name, version=version, sha256=sha256, store=store.entry_path(sha256)
    )


def _submitted_lock(served: _Served) -> str | None:
    _headers, body = served.sent_to(JOBS_URL)
    assert body is not None
    lock = json.loads(body)["lock"]
    assert lock is None or isinstance(lock, str)
    return lock


def test_a_global_only_setup_submits_the_global_entries(
    served: _Served, logged_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    write_lockfile(
        store.global_lock_path(), [_registry_entry("broadcast/tools", "1.0.0", "a" * 64)]
    )
    _submit_accepted(served)
    remote.submit_run(_query(STREAM_QUERY), None, _run_args())
    lock = _submitted_lock(served)
    assert lock is not None
    entries = json.loads(lock)["packages"]
    assert [(one["name"], one["version"], one["sha256"]) for one in entries] == [
        ("broadcast/tools", "1.0.0", "a" * 64)
    ]


def test_a_name_in_both_lockfiles_submits_the_projects_version(
    served: _Served, logged_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    write_lockfile(
        tmp_path / "ffrwd.lock", [_registry_entry("broadcast/tools", "2.0.0", "b" * 64)]
    )
    write_lockfile(
        store.global_lock_path(),
        [
            _registry_entry("broadcast/tools", "1.0.0", "a" * 64),
            _registry_entry("broadcast/extra", "1.0.0", "c" * 64),
        ],
    )
    _submit_accepted(served)
    remote.submit_run(_query(STREAM_QUERY), None, _run_args())
    lock = _submitted_lock(served)
    assert lock is not None
    entries = json.loads(lock)["packages"]
    # The project's pin wins its name; the global-only package still rides.
    assert [(one["name"], one["version"], one["sha256"]) for one in entries] == [
        ("broadcast/tools", "2.0.0", "b" * 64),
        ("broadcast/extra", "1.0.0", "c" * 64),
    ]
    assert "a" * 64 not in lock


def test_a_project_only_lock_travels_verbatim(
    served: _Served, logged_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    write_lockfile(
        tmp_path / "ffrwd.lock", [_registry_entry("broadcast/tools", "2.0.0", "b" * 64)]
    )
    _submit_accepted(served)
    remote.submit_run(_query(STREAM_QUERY), None, _run_args())
    assert _submitted_lock(served) == (tmp_path / "ffrwd.lock").read_text(encoding="utf-8")


def test_no_lockfile_anywhere_submits_none(
    served: _Served, logged_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _submit_accepted(served)
    remote.submit_run(_query(STREAM_QUERY), None, _run_args())
    assert _submitted_lock(served) is None


# ---------------------------------------------------------------------------
# a linked package travels as bytes
# ---------------------------------------------------------------------------


def _linked_package(
    root: Path, *, name: str = "broadcast/tools", version: str = "1.0.0", factor: str = "0.5"
) -> Path:
    """A package directory: a manifest, and one lib file defining ``quieter``."""
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "lib.sql").write_text(
        "CREATE FUNCTION quieter(track audio_stream) RETURNS audio_stream AS $$\n"
        f"  SELECT volume(track, {factor})\n"
        "$$ LANGUAGE sql;\n",
        encoding="utf-8",
    )
    (root / "ffrwd.json").write_text(
        json.dumps({"name": name, "version": version, "lib": {"quieter": "src/lib.sql"}}),
        encoding="utf-8",
    )
    return root


def _submitted_packages(served: _Served) -> list[str]:
    _headers, body = served.sent_to(JOBS_URL)
    assert body is not None
    digests = json.loads(body)["packages"]
    assert isinstance(digests, list)
    return [str(one) for one in digests]


def _uploaded(served: _Served) -> dict[str, bytes]:
    """Every content-addressed POST this submit made, keyed by its digest."""
    sent: dict[str, bytes] = {}
    for url, _headers, body in served.asked:
        if url.startswith(f"{UPLOAD_URL}?sha256=") and body is not None:
            sent[url.rpartition("=")[2]] = body
    return sent


def _packed(root: Path) -> tuple[bytes, str]:
    archive = store.pack(root)
    return archive, hashlib.sha256(archive).hexdigest()


def test_a_linked_package_packs_and_its_digest_pins_the_submitted_lock(
    served: _Served, logged_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    linked = _linked_package(tmp_path / "tools")
    write_lockfile(tmp_path / "ffrwd.lock", [LinkEntry(path="tools")])
    archive, digest = _packed(linked)
    _submit_accepted(served)
    served.answers[f"{UPLOAD_URL}?sha256={digest}"] = json.dumps(
        {"already": False, "bytes": len(archive)}
    ).encode("utf-8")

    remote.submit_run(_query(STREAM_QUERY), None, _run_args())

    # The spec names the digest as one this submit uploads, not one the
    # registry publishes.
    assert _submitted_packages(served) == [digest]
    # The lock the runner reads holds a registry pin, not a link.
    lock = _submitted_lock(served)
    assert lock is not None
    document = json.loads(lock)
    assert document["reproducible"] is True
    assert document["packages"] == [
        {
            "kind": "registry",
            "name": "broadcast/tools",
            "version": "1.0.0",
            "sha256": digest,
            "store": store.entry_path(digest),
        }
    ]
    # And the bytes went to the same content-addressed endpoint an input uses.
    assert _uploaded(served) == {digest: archive}


def test_a_link_recorded_in_the_links_file_packs_the_same_way(
    served: _Served, logged_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where the link is recorded does not change what travels."""
    monkeypatch.chdir(tmp_path)
    linked = _linked_package(tmp_path / "tools")
    write_lockfile(tmp_path / "ffrwd.lock", [])
    write_linksfile(links_path(tmp_path / "ffrwd.lock"), [LinkEntry(path="tools")])
    archive, digest = _packed(linked)
    _submit_accepted(served)
    served.answers[f"{UPLOAD_URL}?sha256={digest}"] = json.dumps(
        {"already": False, "bytes": len(archive)}
    ).encode("utf-8")

    remote.submit_run(_query(STREAM_QUERY), None, _run_args())

    assert _submitted_packages(served) == [digest]
    lock = _submitted_lock(served)
    assert lock is not None
    document = json.loads(lock)
    assert document["reproducible"] is True
    assert [entry["sha256"] for entry in document["packages"]] == [digest]


def test_the_packed_archive_holds_the_manifests_closure(
    served: _Served, logged_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What travels is what `publish` would ship: the closure, and no more."""
    monkeypatch.chdir(tmp_path)
    linked = _linked_package(tmp_path / "tools")
    (linked / "build").mkdir()
    (linked / "build" / "module.wasm").write_bytes(b"x" * 4096)
    (linked / ".ffrwdignore").write_text("build/\n", encoding="utf-8")
    write_lockfile(tmp_path / "ffrwd.lock", [LinkEntry(path="tools")])
    _submit_accepted(served)
    _archive, digest = _packed(linked)
    served.answers[f"{UPLOAD_URL}?sha256={digest}"] = json.dumps({"already": False}).encode(
        "utf-8"
    )

    remote.submit_run(_query(STREAM_QUERY), None, _run_args())

    with tarfile.open(fileobj=io.BytesIO(_uploaded(served)[digest]), mode="r:gz") as opened:
        held = sorted(member.name for member in opened.getmembers() if member.isreg())
    assert held == ["ffrwd.json", "src/lib.sql"]


def test_an_unchanged_link_hits_the_endpoints_dedup_on_a_second_submit(
    served: _Served, logged_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The digest is over the packed bytes, so an unedited link keeps it.

    The endpoint answers ``already`` for content it holds, so a second submit
    stores nothing new; an edit changes the digest and pays again.
    """
    monkeypatch.chdir(tmp_path)
    linked = _linked_package(tmp_path / "tools")
    write_lockfile(tmp_path / "ffrwd.lock", [LinkEntry(path="tools")])
    _archive, digest = _packed(linked)
    _submit_accepted(served)
    served.answers[f"{UPLOAD_URL}?sha256={digest}"] = json.dumps(
        {"already": False, "bytes": 1}
    ).encode("utf-8")
    remote.submit_run(_query(STREAM_QUERY), None, _run_args())
    assert _submitted_packages(served) == [digest]

    # Same tree, same digest -- and the endpoint says it already holds it.
    again = _Served()
    monkeypatch.setattr(packages, "_urlopen", again)
    _submit_accepted(again)
    again.answers[f"{UPLOAD_URL}?sha256={digest}"] = json.dumps(
        {"already": True, "bytes": 1}
    ).encode("utf-8")
    remote.submit_run(_query(STREAM_QUERY), None, _run_args())
    assert _submitted_packages(again) == [digest]

    # An edit is what pays again.
    edited = _Served()
    monkeypatch.setattr(packages, "_urlopen", edited)
    _linked_package(tmp_path / "tools", factor="0.9")
    _archive, changed = _packed(linked)
    _submit_accepted(edited)
    edited.answers[f"{UPLOAD_URL}?sha256={changed}"] = json.dumps({"already": False}).encode(
        "utf-8"
    )
    remote.submit_run(_query(STREAM_QUERY), None, _run_args())
    assert _submitted_packages(edited) == [changed]
    assert changed != digest


def test_a_link_inside_a_linked_package_travels_too(
    served: _Served, logged_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The closure is transitive: a linked package may link others."""
    monkeypatch.chdir(tmp_path)
    tools = _linked_package(tmp_path / "tools")
    helper = _linked_package(tmp_path / "helper", name="broadcast/helper", version="2.1.0")
    # The linked package's OWN lockfile links a second directory.
    write_lockfile(tools / "ffrwd.lock", [LinkEntry(path="../helper")])
    write_lockfile(tmp_path / "ffrwd.lock", [LinkEntry(path="tools")])
    _tools_archive, tools_digest = _packed(tools)
    _helper_archive, helper_digest = _packed(helper)
    _submit_accepted(served)
    for one in (tools_digest, helper_digest):
        served.answers[f"{UPLOAD_URL}?sha256={one}"] = json.dumps({"already": False}).encode(
            "utf-8"
        )

    remote.submit_run(_query(STREAM_QUERY), None, _run_args())

    # Depth first: the package a link names is packed before the link itself.
    assert _submitted_packages(served) == [helper_digest, tools_digest]
    assert sorted(_uploaded(served)) == sorted([helper_digest, tools_digest])
    lock = _submitted_lock(served)
    assert lock is not None
    pinned = [(one["name"], one["version"]) for one in json.loads(lock)["packages"]]
    # The top-level lock pins what it links; the nested one rode along as bytes.
    assert pinned == [("broadcast/tools", "1.0.0")]


def test_a_link_cycle_is_refused_naming_the_loop(
    served: _Served, logged_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    tools = _linked_package(tmp_path / "tools")
    helper = _linked_package(tmp_path / "helper", name="broadcast/helper")
    write_lockfile(tools / "ffrwd.lock", [LinkEntry(path="../helper")])
    write_lockfile(helper / "ffrwd.lock", [LinkEntry(path="../tools")])
    write_lockfile(tmp_path / "ffrwd.lock", [LinkEntry(path="tools")])
    _submit_accepted(served)

    with pytest.raises(FfrwdError) as caught:
        remote.submit_run(_query(STREAM_QUERY), None, _run_args())
    assert caught.value.message == (
        "link cycle: broadcast/tools -> broadcast/helper -> broadcast/tools"
    )
    assert "stop linking another in the loop" in (caught.value.hint or "")
    assert served.asked == []


def test_an_oversize_package_is_refused_naming_the_ignore_file(
    served: _Served, logged_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    linked = _linked_package(tmp_path / "tools")
    (linked / "media.mkv").write_bytes(b"take one" * 512)
    write_lockfile(tmp_path / "ffrwd.lock", [LinkEntry(path="tools")])
    # The real cap is megabytes; what this pins is the refusal, not the number.
    monkeypatch.setattr(remote, "MAX_PACKAGE_BYTES", 64)
    _submit_accepted(served)

    with pytest.raises(FfrwdError) as caught:
        remote.submit_run(_query(STREAM_QUERY), None, _run_args())
    assert caught.value.message.startswith("package 'broadcast/tools' packs to ")
    assert "a submit carries at most 64 bytes per package" in caught.value.message
    assert ".ffrwdignore" in (caught.value.hint or "")
    assert served.asked == []


def test_an_unreadable_link_is_refused(
    served: _Served, logged_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    write_lockfile(tmp_path / "ffrwd.lock", [LinkEntry(path="gone")])
    _submit_accepted(served)

    with pytest.raises(FfrwdError) as caught:
        remote.submit_run(_query(STREAM_QUERY), None, _run_args())
    assert caught.value.message.startswith("the link to gone could not be read:")
    assert "run without --remote" in (caught.value.hint or "")
    assert served.asked == []


def test_a_submit_leaves_the_lockfile_on_disk_alone(
    served: _Served, logged_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A link stays a link locally; the rewrite exists only on the wire."""
    monkeypatch.chdir(tmp_path)
    linked = _linked_package(tmp_path / "tools")
    lock_path = tmp_path / "ffrwd.lock"
    write_lockfile(lock_path, [LinkEntry(path="tools")])
    before = lock_path.read_text(encoding="utf-8")
    _archive, digest = _packed(linked)
    _submit_accepted(served)
    served.answers[f"{UPLOAD_URL}?sha256={digest}"] = json.dumps({"already": False}).encode(
        "utf-8"
    )

    remote.submit_run(_query(STREAM_QUERY), None, _run_args())

    assert lock_path.read_text(encoding="utf-8") == before
    assert json.loads(before)["packages"] == [{"kind": "link", "path": "tools"}]


def test_a_quiet_submit_prints_only_the_result(
    served: _Served,
    logged_in: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "in.mp4").write_bytes(b"media bytes")
    digest = hashlib.sha256(b"media bytes").hexdigest()
    _submit_accepted(served)
    served.answers[f"{UPLOAD_URL}?sha256={digest}"] = json.dumps(
        {"already": False, "bytes": 11}
    ).encode("utf-8")

    code = cli.main(["run", "--remote", "-q", MEDIA_QUERY])
    captured = capsys.readouterr()
    assert code == 0
    assert f"submitted {JOB_ID}" in captured.out
    assert captured.err == ""


def test_an_already_staged_upload_is_a_success(
    served: _Served,
    logged_in: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "in.mp4").write_bytes(b"media bytes")
    digest = hashlib.sha256(b"media bytes").hexdigest()
    _submit_accepted(served)
    served.answers[f"{UPLOAD_URL}?sha256={digest}"] = json.dumps(
        {"already": True, "bytes": 11}
    ).encode("utf-8")

    code = cli.main(["run", "--remote", MEDIA_QUERY])
    out = capsys.readouterr().out
    assert code == 0
    assert f"submitted {JOB_ID}" in out
    start_headers, _body = served.sent_to(START_URL)
    assert start_headers["x-job-token"] == JOB_TOKEN


def test_the_servers_refusal_passes_through_as_itself(
    served: _Served,
    logged_in: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "in.mp4").write_bytes(b"media bytes")
    served.answers[JOBS_URL] = (
        402,
        json.dumps(
            {
                "error": "the free allotment is spent: 0m cpu and 0m gpu left this month",
                "hint": "it resets on 2026-09-01; until then, run without --remote",
            }
        ).encode("utf-8"),
    )
    code = cli.main(["run", "--remote", MEDIA_QUERY])
    err = capsys.readouterr().err
    assert code == 1
    assert "the free allotment is spent: 0m cpu and 0m gpu left this month" in err
    assert "it resets on 2026-09-01; until then, run without --remote" in err


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------

NOW = datetime(2026, 8, 29, 12, 0, 0, tzinfo=timezone.utc)

ROW_DONE = {
    "id": "aaaa1111-0000-0000-0000-000000000000",
    "state": "succeeded",
    "recipe": "broadcast/tracks:duck",
    "gpu": True,
    "duration_cpu_s": 300,
    "duration_gpu_s": 60,
    "created_at": "2026-08-29T11:00:00+00:00",
}
ROW_OLD = {
    "id": "bbbb2222-0000-0000-0000-000000000000",
    "state": "failed",
    "recipe": None,
    "gpu": False,
    "duration_cpu_s": 90,
    "duration_gpu_s": None,
    "created_at": "2026-07-10T00:00:00+00:00",
}


def _listing(
    served: _Served,
    *rows: dict[str, object],
    remaining: dict[str, object] | None = REMAINING,
) -> None:
    body: dict[str, object] = {"jobs": list(rows)}
    if remaining is not None:
        body["remaining"] = remaining
    served.answers[JOBS_URL] = json.dumps(body).encode("utf-8")


@pytest.fixture
def fixed_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(remote, "_utcnow", lambda: NOW)


def test_jobs_lists_a_table_with_the_month_footer(
    served: _Served, logged_in: None, fixed_clock: None, capsys: pytest.CaptureFixture[str]
) -> None:
    _listing(served, ROW_DONE, ROW_OLD)
    code = cli.main(["jobs"])
    out = capsys.readouterr().out
    assert code == 0
    # The cells are this test's expectation; the layout is render_table's,
    # pinned by the cookbook. July's job shows in the table but stays out of
    # the month line: only what this month spent counts.
    expected = TableResult(
        columns=["id", "state", "recipe", "gpu", "runtime", "created"],
        rows=[
            ["aaaa1111", "succeeded", "broadcast/tracks:duck", True, "5m00s+1m00s gpu", "1h ago"],
            ["bbbb2222", "failed", "sql", False, "1m30s", "50d ago"],
        ],
    )
    assert out == render_table(expected) + (
        "\nthis month: 5 min cpu · 1 min gpu · 1 jobs\n"
        "free this month: 47m CPU + 15m GPU remaining\n"
    )


def test_a_listing_with_no_remaining_prints_only_the_month_footer(
    served: _Served, logged_in: None, fixed_clock: None, capsys: pytest.CaptureFixture[str]
) -> None:
    # An older server sends no `remaining` at all -- nothing about it prints.
    _listing(served, ROW_DONE, remaining=None)
    code = cli.main(["jobs"])
    out = capsys.readouterr().out
    assert code == 0
    assert "this month:" in out
    assert "free this month:" not in out


def test_the_free_line_floors_minutes_and_never_goes_negative(
    served: _Served, logged_in: None, fixed_clock: None, capsys: pytest.CaptureFixture[str]
) -> None:
    _listing(served, ROW_DONE, remaining={"cpu_seconds": 119, "gpu_seconds": 59})
    code = cli.main(["jobs"])
    out = capsys.readouterr().out
    assert code == 0
    assert "free this month: 1m CPU + 0m GPU remaining" in out


def test_jobs_json_prints_the_rows_and_remaining_verbatim(
    served: _Served, logged_in: None, capsys: pytest.CaptureFixture[str]
) -> None:
    _listing(served, ROW_DONE, ROW_OLD)
    code = cli.main(["jobs", "--json"])
    captured = capsys.readouterr()
    assert code == 0
    assert json.loads(captured.out) == {"jobs": [ROW_DONE, ROW_OLD], "remaining": REMAINING}
    assert captured.err == ""  # nothing narrates over a machine-read listing


def test_jobs_json_carries_a_null_remaining_when_the_server_sends_none(
    served: _Served, logged_in: None, capsys: pytest.CaptureFixture[str]
) -> None:
    _listing(served, ROW_DONE, remaining=None)
    code = cli.main(["jobs", "--json"])
    captured = capsys.readouterr()
    assert code == 0
    assert json.loads(captured.out) == {"jobs": [ROW_DONE], "remaining": None}


def test_watch_redraws_until_nothing_is_running(
    served: _Served,
    logged_in: None,
    fixed_clock: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    running = dict(ROW_DONE, state="running")
    _listing(served, running)
    slept: list[float] = []

    def _finish(seconds: float) -> None:
        slept.append(seconds)
        _listing(served, ROW_DONE)

    monkeypatch.setattr(remote, "_sleep", _finish)
    code = cli.main(["jobs", "--watch"])
    out = capsys.readouterr().out
    assert code == 0
    assert slept == [remote.WATCH_SECONDS]
    assert out.count("\nthis month:") == 2
    assert out.count("free this month:") == 2  # both footer lines redraw
    assert "running" in out and "succeeded" in out


def test_cancel_resolves_a_prefix_against_the_callers_jobs(
    served: _Served, logged_in: None, capsys: pytest.CaptureFixture[str]
) -> None:
    _listing(served, ROW_DONE, ROW_OLD)
    cancel_url = f"{JOBS_URL}/{ROW_OLD['id']}/cancel"
    served.answers[cancel_url] = json.dumps(dict(ROW_OLD, state="cancelled")).encode(
        "utf-8"
    )
    code = cli.main(["jobs", "--cancel", "bbbb"])
    out = capsys.readouterr().out
    assert code == 0
    assert "cancelled bbbb2222" in out
    headers, body = served.sent_to(cancel_url)
    assert headers["authorization"] == f"Bearer {TOKEN}"
    assert body == b"{}"


def test_an_ambiguous_prefix_names_the_matches(
    served: _Served, logged_in: None, capsys: pytest.CaptureFixture[str]
) -> None:
    twin = dict(ROW_OLD, id="aaaa9999-0000-0000-0000-000000000000")
    _listing(served, ROW_DONE, twin)
    code = cli.main(["jobs", "--cancel", "aaaa"])
    err = capsys.readouterr().err
    assert code == 1
    assert "'aaaa' matches more than one job" in err
    assert str(ROW_DONE["id"]) in err and str(twin["id"]) in err
    assert "give more of the id" in err


def test_a_prefix_matching_nothing_says_so(
    served: _Served, logged_in: None, capsys: pytest.CaptureFixture[str]
) -> None:
    _listing(served, ROW_DONE)
    code = cli.main(["jobs", "--cancel", "zzzz"])
    err = capsys.readouterr().err
    assert code == 1
    assert "'zzzz' matches none of your jobs" in err


# ---------------------------------------------------------------------------
# jobs <id>
# ---------------------------------------------------------------------------


def _detail_url(job_id: str) -> str:
    return f"{JOBS_URL}/{job_id}"


def test_jobs_id_shows_one_jobs_detail_including_the_tail(
    served: _Served, logged_in: None, fixed_clock: None, capsys: pytest.CaptureFixture[str]
) -> None:
    _listing(served, ROW_DONE, ROW_OLD)
    detail = dict(
        ROW_DONE,
        log_tail="frame=1\nframe=2\nframe=3",
        outputs=[
            {"path": "clips/out.mp4", "bytes": 123, "sha256": "a" * 64},
            {
                "path": "ladder/master.m3u8",
                "bytes": 456,
                "sha256": "b" * 64,
                "kind": "tree",
                "files": 42,
            },
        ],
    )
    served.answers[_detail_url(str(ROW_DONE["id"]))] = json.dumps(detail).encode("utf-8")
    code = cli.main(["jobs", "aaaa"])
    out = capsys.readouterr().out
    assert code == 0
    # The listing's own row rendering, plus what a listing never carries.
    assert "aaaa1111" in out and "succeeded" in out
    assert "output: clips/out.mp4" in out
    assert "output: ladder/master.m3u8 (42 files)" in out
    assert "log tail:" in out
    assert "frame=1" in out and "frame=3" in out


def test_jobs_id_resolves_a_unique_prefix(
    served: _Served, logged_in: None, fixed_clock: None, capsys: pytest.CaptureFixture[str]
) -> None:
    _listing(served, ROW_DONE, ROW_OLD)
    served.answers[_detail_url(str(ROW_OLD["id"]))] = json.dumps(
        dict(ROW_OLD, error="ffmpeg exited 1", hint="check the filter chain")
    ).encode("utf-8")
    code = cli.main(["jobs", "bbbb"])
    out = capsys.readouterr().out
    assert code == 0
    assert "bbbb2222" in out and "failed" in out
    assert "error: ffmpeg exited 1" in out
    assert "hint: check the filter chain" in out


def test_jobs_id_with_a_null_tail_prints_nothing_extra(
    served: _Served, logged_in: None, fixed_clock: None, capsys: pytest.CaptureFixture[str]
) -> None:
    _listing(served, ROW_DONE)
    detail = dict(ROW_DONE, log_tail=None, outputs=None)
    served.answers[_detail_url(str(ROW_DONE["id"]))] = json.dumps(detail).encode("utf-8")
    code = cli.main(["jobs", "aaaa"])
    out = capsys.readouterr().out
    assert code == 0
    assert "log tail" not in out
    assert "output:" not in out


def test_jobs_id_truncates_a_long_tail_and_says_so(
    served: _Served, logged_in: None, fixed_clock: None, capsys: pytest.CaptureFixture[str]
) -> None:
    _listing(served, ROW_DONE)
    lines = [f"L{n:02d}" for n in range(1, 26)]  # 25 lines; the cap is 20
    detail = dict(ROW_DONE, log_tail="\n".join(lines), outputs=None)
    served.answers[_detail_url(str(ROW_DONE["id"]))] = json.dumps(detail).encode("utf-8")
    code = cli.main(["jobs", "aaaa"])
    out = capsys.readouterr().out
    assert code == 0
    assert "log tail (last 20 of 25 lines, 5 cut):" in out
    assert "L01" not in out and "L05" not in out  # the earliest 5 are cut
    assert "L06" in out and "L25" in out  # the last 20 remain


def test_jobs_id_json_carries_the_log_tail_verbatim(
    served: _Served, logged_in: None, capsys: pytest.CaptureFixture[str]
) -> None:
    _listing(served, ROW_DONE)
    detail = dict(ROW_DONE, log_tail="line one\nline two", outputs=None)
    served.answers[_detail_url(str(ROW_DONE["id"]))] = json.dumps(detail).encode("utf-8")
    code = cli.main(["jobs", "aaaa", "--json"])
    captured = capsys.readouterr()
    assert code == 0
    assert json.loads(captured.out) == detail
    assert captured.err == ""


def test_jobs_id_does_not_mix_with_watch_cancel_or_fetch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = cli.main(["jobs", "aaaa", "--watch"])
    err = capsys.readouterr().err
    assert code == 2
    assert "an ID does not mix with --watch/--cancel/--fetch" in err


def _fetchable(
    served: _Served,
    content: bytes,
    path: str = "clips/out.mp4",
    *,
    kind: str | None = None,
    files: int | None = None,
) -> str:
    """One fetchable output over `content`'s digest. No `kind` key unless
    given: an older row's entry, which fetch treats as a file."""
    download_url = "https://runner.example/download/aaaa/0?token=t"
    fetch_url = f"{JOBS_URL}/{ROW_DONE['id']}/fetch"
    _listing(served, ROW_DONE)
    entry: dict[str, object] = {
        "path": path,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "url": download_url,
    }
    if kind is not None:
        entry["kind"] = kind
    if files is not None:
        entry["files"] = files
    served.answers[fetch_url] = json.dumps(
        {"job_id": ROW_DONE["id"], "outputs": [entry]}
    ).encode("utf-8")
    return download_url


def _tree_tar(members: dict[str, bytes | None]) -> bytes:
    """An uncompressed tar of `members`: bytes for a file, None for a directory."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            if content is None:
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
                continue
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return raw.getvalue()


def _symlink_tar() -> bytes:
    """An archive whose one member is a symlink, which no runner packs."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        info = tarfile.TarInfo("master.m3u8")
        info.type = tarfile.SYMTYPE
        info.linkname = "../outside"
        archive.addfile(info)
    return raw.getvalue()


def test_fetch_streams_each_output_to_its_as_written_path(
    served: _Served,
    logged_in: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    download_url = _fetchable(served, b"the output bytes")
    served.answers[download_url] = b"the output bytes"
    code = cli.main(["jobs", "--fetch", "aaaa"])
    captured = capsys.readouterr()
    assert code == 0
    assert "wrote clips/out.mp4" in captured.out
    assert "downloading clips/out.mp4 (16 bytes)\n" in captured.err
    assert (tmp_path / "clips" / "out.mp4").read_bytes() == b"the output bytes"


def test_fetch_refuses_an_existing_file_without_dash_y(
    served: _Served,
    logged_in: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    download_url = _fetchable(served, b"new bytes")
    served.answers[download_url] = b"new bytes"
    (tmp_path / "clips").mkdir()
    (tmp_path / "clips" / "out.mp4").write_bytes(b"old bytes")

    code = cli.main(["jobs", "--fetch", "aaaa"])
    err = capsys.readouterr().err
    assert code == 1
    assert "'clips/out.mp4' already exists" in err
    assert "pass -y to overwrite it" in err
    assert (tmp_path / "clips" / "out.mp4").read_bytes() == b"old bytes"

    code = cli.main(["jobs", "--fetch", "aaaa", "-y"])
    assert code == 0
    assert (tmp_path / "clips" / "out.mp4").read_bytes() == b"new bytes"


def test_fetch_verifies_the_digest_and_leaves_no_file(
    served: _Served,
    logged_in: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    download_url = _fetchable(served, b"what the job recorded")
    served.answers[download_url] = b"something else entirely"
    code = cli.main(["jobs", "--fetch", "aaaa"])
    err = capsys.readouterr().err
    assert code == 1
    assert "downloaded as" in err and "the job recorded" in err
    assert not (tmp_path / "clips" / "out.mp4").exists()
    assert list((tmp_path / "clips").glob("*.part")) == []


def test_fetch_streams_an_explicit_file_kind_to_its_path(
    served: _Served,
    logged_in: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    download_url = _fetchable(served, b"the output bytes", kind="file")
    served.answers[download_url] = b"the output bytes"
    code = cli.main(["jobs", "--fetch", "aaaa"])
    assert code == 0
    assert (tmp_path / "clips" / "out.mp4").read_bytes() == b"the output bytes"


TREE_PATH = "ladder/master.m3u8"


def test_fetch_unpacks_a_tree_at_its_as_written_path(
    served: _Served,
    logged_in: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    tar = _tree_tar(
        {
            "master.m3u8": b"#EXTM3U master",
            "v0": None,
            "v0/rung.m3u8": b"#EXTM3U rung",
            "v0/seg_000.ts": b"segment bytes",
        }
    )
    download_url = _fetchable(served, tar, TREE_PATH, kind="tree", files=3)
    served.answers[download_url] = tar
    code = cli.main(["jobs", "--fetch", "aaaa"])
    captured = capsys.readouterr()
    assert code == 0
    assert "wrote ladder/master.m3u8" in captured.out
    assert (tmp_path / "ladder" / "master.m3u8").read_bytes() == b"#EXTM3U master"
    assert (tmp_path / "ladder" / "v0" / "rung.m3u8").read_bytes() == b"#EXTM3U rung"
    assert (tmp_path / "ladder" / "v0" / "seg_000.ts").read_bytes() == b"segment bytes"
    # The archive is wire format: neither it nor any staging survives.
    landed = sorted(one.relative_to(tmp_path).as_posix() for one in tmp_path.rglob("*"))
    assert landed == [
        "ladder",
        "ladder/master.m3u8",
        "ladder/v0",
        "ladder/v0/rung.m3u8",
        "ladder/v0/seg_000.ts",
    ]


def test_fetch_verifies_a_tree_archives_digest_and_leaves_nothing(
    served: _Served,
    logged_in: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    tar = _tree_tar({"master.m3u8": b"#EXTM3U"})
    download_url = _fetchable(served, tar, TREE_PATH, kind="tree", files=1)
    served.answers[download_url] = b"not the archive the job recorded"
    code = cli.main(["jobs", "--fetch", "aaaa"])
    err = capsys.readouterr().err
    assert code == 1
    assert "'ladder/master.m3u8' downloaded as" in err
    assert list(tmp_path.rglob("*")) == []


@pytest.mark.parametrize(
    ("archive", "said"),
    [
        pytest.param(
            _tree_tar({"../escape.ts": b"x"}),
            "leaves the directory it unpacks into",
            id="dotdot",
        ),
        pytest.param(
            _tree_tar({"/escape.ts": b"x"}),
            "leaves the directory it unpacks into",
            id="absolute",
        ),
        pytest.param(
            _symlink_tar(), "neither a plain file nor a directory", id="symlink"
        ),
    ],
)
def test_fetch_refuses_a_hostile_tree_member_and_leaves_nothing(
    served: _Served,
    logged_in: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    archive: bytes,
    said: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    download_url = _fetchable(served, archive, TREE_PATH, kind="tree", files=1)
    served.answers[download_url] = archive
    code = cli.main(["jobs", "--fetch", "aaaa"])
    err = capsys.readouterr().err
    assert code == 1
    assert said in err
    assert "nothing was written" in err
    assert list(tmp_path.rglob("*")) == []


def test_fetch_refuses_a_tree_into_a_nonempty_directory_without_dash_y(
    served: _Served,
    logged_in: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    tar = _tree_tar({"master.m3u8": b"new master", "seg_000.ts": b"new segment"})
    download_url = _fetchable(served, tar, TREE_PATH, kind="tree", files=2)
    served.answers[download_url] = tar
    (tmp_path / "ladder").mkdir()
    (tmp_path / "ladder" / "unrelated.txt").write_bytes(b"kept")

    code = cli.main(["jobs", "--fetch", "aaaa"])
    err = capsys.readouterr().err
    assert code == 1
    assert "'ladder/master.m3u8' unpacks a tree into 'ladder', which is not empty" in err
    assert "pass -y to overwrite the files the tree carries" in err
    assert not (tmp_path / "ladder" / "master.m3u8").exists()

    # -y overwrites what the tree carries and leaves the rest alone.
    (tmp_path / "ladder" / "master.m3u8").write_bytes(b"old master")
    code = cli.main(["jobs", "--fetch", "aaaa", "-y"])
    assert code == 0
    assert (tmp_path / "ladder" / "master.m3u8").read_bytes() == b"new master"
    assert (tmp_path / "ladder" / "seg_000.ts").read_bytes() == b"new segment"
    assert (tmp_path / "ladder" / "unrelated.txt").read_bytes() == b"kept"


def test_fetch_refuses_a_manifest_path_leaving_the_working_directory(
    served: _Served,
    logged_in: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    for path in ("/etc/evil.mp4", "../evil.mp4", "C:\\evil.mp4"):
        download_url = _fetchable(served, b"payload", path)
        served.answers[download_url] = b"payload"
        code = cli.main(["jobs", "--fetch", "aaaa"])
        err = capsys.readouterr().err
        assert code == 1
        assert (
            f"the job recorded output '{path}', which does not stay under the "
            "working directory" in err
        )
        assert list(tmp_path.rglob("*")) == []
        # Refused before the first byte: the download URL was never asked for.
        assert download_url not in [asked for asked, _headers, _body in served.asked]


def test_jobs_without_a_token_is_refused(
    served: _Served, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["jobs"])
    err = capsys.readouterr().err
    assert code == 1
    assert "running remotely needs an ffrwd token" in err
    assert served.asked == []


# ---------------------------------------------------------------------------
# run --remote --wait
# ---------------------------------------------------------------------------

# No local file, so submit needs no upload: --wait's own tests are about the
# poll and the fetch, not the submit machinery `test_submit_...` already covers.
WAIT_QUERY = "COPY (SELECT a.video[1] FROM input('https://cdn.example/x.mp4') a) TO 'out.mp4'"

DOWNLOAD_URL = "https://runner.example/download/aaaa/0?token=t"


def _wait_row(state: str, *, progress_pct: float = 0) -> dict[str, object]:
    return {
        "id": JOB_ID,
        "state": state,
        "progress_pct": progress_pct,
        "recipe": None,
        "gpu": False,
        "duration_cpu_s": 0,
        "duration_gpu_s": None,
        "created_at": "2026-08-29T11:00:00+00:00",
    }


def _listing_body(
    *rows: dict[str, object], remaining: dict[str, object] | None = REMAINING
) -> dict[str, object]:
    body: dict[str, object] = {"jobs": list(rows)}
    if remaining is not None:
        body["remaining"] = remaining
    return body


def _wait_polling(served: _Served, url: str, bodies: list[dict[str, object]]) -> object:
    """A GET-`url` sequence for --wait's poll: each of `bodies` in turn, the
    last one repeating past the end. Submit's own POST to the same URL still
    answers from `served.answers`, exactly as every other test's does --
    this only stands in for the listing GETs the poll loop makes."""
    calls = {"n": 0}
    base = served.__call__

    def wrapped(request: object, timeout: float | None = None) -> _Fake:
        full = str(request.full_url)  # type: ignore[attr-defined]
        data = request.data  # type: ignore[attr-defined]
        if full == url and data is None:
            index = min(calls["n"], len(bodies) - 1)
            calls["n"] += 1
            return _Fake(200, json.dumps(bodies[index]).encode("utf-8"))
        return base(request, timeout=timeout)

    return wrapped


def _wait_fetchable(served: _Served, content: bytes, path: str = "out.mp4") -> None:
    fetch_url = f"{JOBS_URL}/{JOB_ID}/fetch"
    served.answers[fetch_url] = json.dumps(
        {
            "job_id": JOB_ID,
            "outputs": [
                {
                    "path": path,
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "url": DOWNLOAD_URL,
                }
            ],
        }
    ).encode("utf-8")
    served.answers[DOWNLOAD_URL] = content


def test_bar_line_renders_the_fill_and_the_state() -> None:
    line = remote._bar_line({"progress_pct": 42, "state": "running"})
    assert line.startswith("[") and line.endswith("42% running")
    assert line.count("#") == 10  # round(24 * 42 / 100)
    assert line.count("-") == 14


def test_wait_polls_to_success_and_fetches_the_output(
    served: _Served,
    logged_in: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _submit_accepted(served)
    slept: list[float] = []
    monkeypatch.setattr(remote, "_sleep", lambda seconds: slept.append(seconds))
    monkeypatch.setattr(
        packages,
        "_urlopen",
        _wait_polling(
            served,
            JOBS_URL,
            [
                _listing_body(_wait_row("running", progress_pct=40)),
                _listing_body(_wait_row("succeeded", progress_pct=100)),
            ],
        ),
    )
    _wait_fetchable(served, b"the output bytes")

    code = cli.main(["run", "--remote", "--wait", WAIT_QUERY])
    captured = capsys.readouterr()
    assert code == 0
    assert "wrote out.mp4" in captured.out
    assert (tmp_path / "out.mp4").read_bytes() == b"the output bytes"
    assert slept == [remote.WATCH_SECONDS]


def test_wait_on_failure_prints_the_rows_error_and_exits_1(
    served: _Served,
    logged_in: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _submit_accepted(served)
    monkeypatch.setattr(remote, "_sleep", lambda seconds: None)
    failed = dict(_wait_row("failed"), error="ffmpeg exited 1: no such filter 'zscale'")
    monkeypatch.setattr(
        packages, "_urlopen", _wait_polling(served, JOBS_URL, [_listing_body(failed)])
    )
    code = cli.main(["run", "--remote", "--wait", WAIT_QUERY])
    err = capsys.readouterr().err
    assert code == 1
    assert "ffmpeg exited 1: no such filter 'zscale'" in err


def test_wait_on_cancelled_prints_the_rows_error_and_exits_1(
    served: _Served,
    logged_in: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _submit_accepted(served)
    monkeypatch.setattr(remote, "_sleep", lambda seconds: None)
    cancelled = dict(_wait_row("cancelled"), error="cancelled by another session")
    monkeypatch.setattr(
        packages, "_urlopen", _wait_polling(served, JOBS_URL, [_listing_body(cancelled)])
    )
    code = cli.main(["run", "--remote", "--wait", WAIT_QUERY])
    err = capsys.readouterr().err
    assert code == 1
    assert "cancelled by another session" in err


def test_wait_on_failure_also_prints_the_tail_beneath_the_error(
    served: _Served,
    logged_in: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _submit_accepted(served)
    monkeypatch.setattr(remote, "_sleep", lambda seconds: None)
    failed = dict(_wait_row("failed"), error="ffmpeg exited 1: no such filter 'zscale'")
    monkeypatch.setattr(
        packages, "_urlopen", _wait_polling(served, JOBS_URL, [_listing_body(failed)])
    )
    served.answers[_detail_url(JOB_ID)] = json.dumps(
        dict(failed, log_tail="frame=100\nerror applying filter\nfailed")
    ).encode("utf-8")
    code = cli.main(["run", "--remote", "--wait", WAIT_QUERY])
    err = capsys.readouterr().err
    assert code == 1
    error_at = err.index("ffmpeg exited 1: no such filter 'zscale'")
    tail_at = err.index("log tail:")
    assert tail_at > error_at  # the tail prints after the error, not before
    assert "frame=100" in err and "error applying filter" in err


def test_wait_json_on_failure_carries_the_log_tail(
    served: _Served,
    logged_in: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _submit_accepted(served)
    monkeypatch.setattr(remote, "_sleep", lambda seconds: None)
    failed = dict(_wait_row("failed"), error="ffmpeg exited 1")
    monkeypatch.setattr(
        packages, "_urlopen", _wait_polling(served, JOBS_URL, [_listing_body(failed)])
    )
    served.answers[_detail_url(JOB_ID)] = json.dumps(dict(failed, log_tail="boom")).encode(
        "utf-8"
    )
    code = cli.main(["run", "--remote", "--wait", "--json", WAIT_QUERY])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["job"]["log_tail"] == "boom"


def test_wait_on_failure_with_no_tail_stored_prints_nothing_extra(
    served: _Served,
    logged_in: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _submit_accepted(served)
    monkeypatch.setattr(remote, "_sleep", lambda seconds: None)
    failed = dict(_wait_row("failed"), error="ffmpeg exited 1")
    monkeypatch.setattr(
        packages, "_urlopen", _wait_polling(served, JOBS_URL, [_listing_body(failed)])
    )
    served.answers[_detail_url(JOB_ID)] = json.dumps(dict(failed, log_tail=None)).encode(
        "utf-8"
    )
    code = cli.main(["run", "--remote", "--wait", WAIT_QUERY])
    err = capsys.readouterr().err
    assert code == 1
    assert "ffmpeg exited 1" in err
    assert "log tail" not in err


def test_watch_prints_a_failed_jobs_error_and_tail_once_it_stops(
    served: _Served,
    logged_in: None,
    fixed_clock: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    running = dict(ROW_OLD, state="running")
    _listing(served, running)
    failed_row = dict(ROW_OLD, error="ffmpeg exited 1")

    def _finish(seconds: float) -> None:
        _listing(served, failed_row)

    monkeypatch.setattr(remote, "_sleep", _finish)
    served.answers[_detail_url(str(ROW_OLD["id"]))] = json.dumps(
        dict(failed_row, log_tail="oops")
    ).encode("utf-8")
    code = cli.main(["jobs", "--watch"])
    captured = capsys.readouterr()
    assert code == 0
    assert "ffmpeg exited 1" in captured.err
    assert "log tail:" in captured.err and "oops" in captured.err


def test_wait_on_budget_exhausted_fetches_and_notes_the_reset(
    served: _Served,
    logged_in: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _submit_accepted(served)
    monkeypatch.setattr(remote, "_sleep", lambda seconds: None)
    # A budget stop is a SUCCESS: state succeeded, budget_exhausted marking it.
    stopped = _wait_row("succeeded", progress_pct=100)
    stopped["budget_exhausted"] = True
    monkeypatch.setattr(
        packages,
        "_urlopen",
        _wait_polling(served, JOBS_URL, [_listing_body(stopped)]),
    )
    _wait_fetchable(served, b"partial output")

    code = cli.main(["run", "--remote", "--wait", WAIT_QUERY])
    out = capsys.readouterr().out
    assert code == 0
    assert "wrote out.mp4" in out
    assert "the free allotment stopped this job early" in out
    assert REMAINING["resets_on"] in out  # type: ignore[operator]


def test_wait_interrupted_detaches_without_cancelling(
    served: _Served,
    logged_in: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _submit_accepted(served)

    def _interrupt(seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(remote, "_sleep", _interrupt)
    monkeypatch.setattr(
        packages,
        "_urlopen",
        _wait_polling(served, JOBS_URL, [_listing_body(_wait_row("running", progress_pct=10))]),
    )
    code = cli.main(["run", "--remote", "--wait", WAIT_QUERY])
    err = capsys.readouterr().err
    assert code == 130
    assert f"ffrwd jobs --watch {JOB_ID[:8]}" in err
    assert f"ffrwd jobs --fetch {JOB_ID[:8]}" in err
    assert "keeps running" in err
    assert all(f"/{JOB_ID}/cancel" not in asked for asked, _headers, _body in served.asked)


def test_wait_without_remote_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["run", "--wait", WAIT_QUERY])
    err = capsys.readouterr().err
    assert code == 2
    assert "error: run: --wait needs --remote" in err


def test_json_with_wait_emits_the_row_and_written_paths(
    served: _Served,
    logged_in: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _submit_accepted(served)
    monkeypatch.setattr(remote, "_sleep", lambda seconds: None)
    monkeypatch.setattr(
        packages,
        "_urlopen",
        _wait_polling(
            served, JOBS_URL, [_listing_body(_wait_row("succeeded", progress_pct=100))]
        ),
    )
    _wait_fetchable(served, b"json path bytes")

    code = cli.main(["run", "--remote", "--wait", "--json", WAIT_QUERY])
    captured = capsys.readouterr()
    assert code == 0
    assert "submitted" not in captured.out  # --json speaks JSON alone on stdout
    payload = json.loads(captured.out)
    assert payload["job"]["id"] == JOB_ID
    assert payload["job"]["state"] == "succeeded"
    assert payload["written"] == ["out.mp4"]


def test_json_with_wait_on_failure_emits_the_row_and_exits_1(
    served: _Served,
    logged_in: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _submit_accepted(served)
    monkeypatch.setattr(remote, "_sleep", lambda seconds: None)
    failed = dict(_wait_row("failed"), error="ffmpeg exited 1")
    monkeypatch.setattr(
        packages, "_urlopen", _wait_polling(served, JOBS_URL, [_listing_body(failed)])
    )
    code = cli.main(["run", "--remote", "--wait", "--json", WAIT_QUERY])
    captured = capsys.readouterr()
    assert code == 1
    payload = json.loads(captured.out)
    assert payload["job"]["state"] == "failed"
    assert payload["job"]["error"] == "ffmpeg exited 1"
