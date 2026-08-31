"""Tests for the package registry client: search, resolve, fetch and install.

No test here opens a socket. Two seams stand in for the two ways a registry
answers.

A LOCAL registry is a directory under ``tmp_path`` holding exactly what a
published one serves off its storage -- ``p/<namespace>/<package>.json`` and
``archives/<sha256>`` -- with ``FFRWD_REGISTRY`` pointing the client at it.
That is a real registry, not a stub: it is what an offline install reads and
what the in-repo package corpus is built against, so the whole install path
runs through it unchanged.

A HOSTED registry is the module's own HTTP seam, ``packages._urlopen``,
replaced by :func:`_served` with a table of URL to answer. Everything only the
hosted one has -- the search function, the archive-signing endpoint, the
private detail endpoint, and the model fetch from the hub -- is exercised
through it, and the URLs it is asked for are themselves the assertion.

The store and the credentials file live under this test's own directory (the
``store_home`` fixture and the suite-wide isolation in ``conftest``), so
nothing leaks between tests or into a developer's home.

The headline check is :func:`test_init_then_install_then_a_query_calling_it`:
the whole loop, from an empty directory to a compiled ffmpeg command that
carries the installed package's own function body.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from ffrwd import cli, credentials, packages, store
from ffrwd.errors import FfrwdError
from ffrwd.mcp import tools as mcp_tools
from ffrwd.project import (
    LinkEntry,
    ModelPin,
    RegistryEntry,
    read_lockfile,
    read_manifest,
    write_lockfile,
)

QUERY = "COPY (SELECT {call}(f.audio[1]) FROM input('film.mkv') f) TO 'out.mkv'"


def _quieter(factor: str) -> str:
    """A one-argument ``quieter`` whose factor shows up in the compiled graph."""
    return (
        "CREATE FUNCTION quieter(track audio_stream) RETURNS audio_stream AS $$\n"
        f"  SELECT volume(track, {factor})\n"
        "$$ LANGUAGE sql;\n"
    )


def _package(
    root: Path,
    *,
    name: str = "broadcast/tracks",
    version: str = "1.0.0",
    factor: str = "0.5",
    description: str = "audio track tools",
    member: str = "quieter",
    src: str | None = None,
    dependencies: dict[str, str] | None = None,
) -> Path:
    """A package directory: a manifest, and one lib file defining ``quieter``.

    `src` overrides the body and `member` its exported name, for a package
    whose own body calls into another package; `dependencies` is its own
    manifest's.
    """
    (root / "src").mkdir(parents=True, exist_ok=True)
    body = src if src is not None else _quieter(factor)
    (root / "src" / "lib.sql").write_text(body, encoding="utf-8")
    declared: dict[str, object] = {
        "name": name,
        "version": version,
        "description": description,
        "lib": {member: "src/lib.sql"},
    }
    if dependencies:
        declared["dependencies"] = dependencies
    (root / "ffrwd.json").write_text(json.dumps(declared, indent=2) + "\n", encoding="utf-8")
    return root


def _read_json(path: Path) -> dict[str, object]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _publish(
    registry: Path,
    source: Path,
    *,
    functions: tuple[str, ...] = ("quieter",),
    recipes: tuple[str, ...] = (),
) -> str:
    """Publish `source` into the fixture registry the way the real one materializes it.

    Writes the two files a client reads -- the archive, and the package's
    detail document with this version appended -- and returns the archive's
    digest. The version entry carries what `ffrwd publish` derives and what
    search reads back off it.
    """
    package = read_manifest(source / "ffrwd.json")
    archive = store.pack(source)
    sha256 = hashlib.sha256(archive).hexdigest()
    (registry / "archives").mkdir(parents=True, exist_ok=True)
    (registry / "archives" / sha256).write_bytes(archive)

    detail = registry / "p" / f"{package.name}.json"
    held: dict[str, object] = (
        _read_json(detail)
        if detail.is_file()
        else {"format_version": 2, "name": package.name, "versions": []}
    )
    versions = list(held["versions"]) if isinstance(held.get("versions"), list) else []
    versions.append(
        {
            "version": package.version,
            "sha256": sha256,
            "size": len(archive),
            "namespace": package.namespace,
            "description": _read_json(source / "ffrwd.json").get("description", ""),
            "functions": [{"name": name} for name in functions],
            "recipes": [{"name": name} for name in recipes],
        }
    )
    held["versions"] = versions
    _write_json(detail, held)
    return sha256


@pytest.fixture
def store_home(_isolated_store: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the store, the machine-wide lockfile and the index cache here."""
    home = tmp_path / "cache"
    home.mkdir()
    monkeypatch.setattr(store, "_cache_dir", lambda: home)
    return home


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty fixture registry, and the environment pointed at it."""
    root = tmp_path / "registry"
    root.mkdir()
    monkeypatch.setenv(packages.REGISTRY_ENV, str(root))
    return root


# ---------------------------------------------------------------------------
# the hosted registry: one seam, a table of answers
# ---------------------------------------------------------------------------

STORAGE = "https://registry.example/packages"
API = "https://api.example"
TOKEN = "ffrwd_" + "A" * 43


class _Fake:
    """One canned HTTP answer, shaped the way urllib hands one back."""

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
    """The HTTP seam: a table of URL to answer, and what was asked for.

    An answer is bytes for a 200, or an ``(status, body)`` pair for anything
    else -- which urllib raises rather than returns, so this does too. A URL
    the table does not hold is a 404, since that is what a registry says about
    something it has never published.
    """

    def __init__(self, answers: Mapping[str, object]) -> None:
        self.answers = dict(answers)
        self.asked: list[tuple[str, dict[str, str], bytes | None]] = []

    def __call__(self, request: object, timeout: float | None = None) -> _Fake:
        assert hasattr(request, "full_url")
        url = str(request.full_url)  # type: ignore[attr-defined]
        headers = {
            str(key).lower(): str(value)
            for key, value in dict(request.headers).items()  # type: ignore[attr-defined]
        }
        body = request.data  # type: ignore[attr-defined]
        self.asked.append((url, headers, body))
        answer = self.answers.get(url, (404, b'{"error": "not found"}'))
        status, content = answer if isinstance(answer, tuple) else (200, answer)
        assert isinstance(content, bytes)
        if status >= 400:
            raise urllib.error.HTTPError(url, status, "refused", {}, io.BytesIO(content))  # type: ignore[arg-type]
        return _Fake(status, content)

    def urls(self) -> list[str]:
        return [url for url, _headers, _body in self.asked]

    def headers_for(self, url: str) -> dict[str, str]:
        for asked, headers, _body in self.asked:
            if asked == url:
                return headers
        raise AssertionError(f"nothing asked for {url}; asked {self.urls()}")

    def body_for(self, url: str) -> bytes:
        for asked, _headers, body in self.asked:
            if asked == url:
                assert body is not None
                return body
        raise AssertionError(f"nothing asked for {url}; asked {self.urls()}")


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Served]:
    """A hosted registry: the two base URLs, and the HTTP seam replaced."""
    monkeypatch.setenv(packages.REGISTRY_ENV, STORAGE)
    monkeypatch.setenv(packages.API_ENV, API)
    fake = _Served({})
    monkeypatch.setattr(packages, "_urlopen", fake)
    yield fake


def _serves(fake: _Served, **answers: object) -> None:
    """Add answers to a served registry, keyed by the constant naming each URL."""
    fake.answers.update(answers)


def _detail_document(name: str, versions: list[dict[str, object]]) -> bytes:
    return json.dumps(
        {"format_version": 2, "name": name, "versions": versions}, indent=2
    ).encode("utf-8")


def _version(sha256: str, size: int, version: str = "1.0.0") -> dict[str, object]:
    return {"version": version, "sha256": sha256, "size": size}


def _detail_url(name: str) -> str:
    return f"{STORAGE}/p/{name}.json"


def _sign_url(sha256: str) -> str:
    return f"{API}/functions/v1/archive/{sha256}"


SEARCH_URL = f"{API}/rest/v1/rpc/search_packages"


def _run(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *argv: str,
) -> tuple[int, str, str]:
    """Run the CLI with `root` as the working directory."""
    monkeypatch.chdir(root)
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _project(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> Path:
    """A project started the way a user starts one."""
    root.mkdir(parents=True, exist_ok=True)
    assert _run(root, monkeypatch, capsys, "init", "--namespace", "consumer")[0] == 0
    return root


# ---------------------------------------------------------------------------
# the base URL
# ---------------------------------------------------------------------------


def test_the_defaults_are_read_when_the_environment_names_neither(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(packages.REGISTRY_ENV, raising=False)
    monkeypatch.delenv(packages.API_ENV, raising=False)
    assert packages.base_url() == packages.DEFAULT_REGISTRY
    assert packages.api_url() == packages.DEFAULT_API
    assert packages.DEFAULT_REGISTRY.startswith("https://")
    assert packages.DEFAULT_API.startswith("https://")


def test_the_environment_overrides_both_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(packages.REGISTRY_ENV, "https://packages.example/registry ")
    monkeypatch.setenv(packages.API_ENV, " https://api.example ")
    assert packages.base_url() == "https://packages.example/registry"
    assert packages.api_url() == "https://api.example"


def test_a_file_url_reads_the_directory_it_names(
    store_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "registry"
    root.mkdir()
    _publish(root, _package(tmp_path / "built"))
    monkeypatch.setenv(packages.REGISTRY_ENV, root.as_uri())
    assert packages.resolve("broadcast/tracks").version == "1.0.0"


def test_a_registry_directory_that_is_not_there_is_a_typed_error(
    store_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typo must not read as a registry publishing nothing."""
    monkeypatch.setenv(packages.REGISTRY_ENV, str(tmp_path / "nowhere"))
    with pytest.raises(FfrwdError) as caught:
        packages.search("tracks")
    assert "there is no such directory" in caught.value.message


# ---------------------------------------------------------------------------
# the whole loop
# ---------------------------------------------------------------------------


def test_init_then_install_then_a_query_calling_it(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _publish(registry, _package(tmp_path / "built"))
    project = _project(tmp_path / "work", monkeypatch, capsys)

    code, out, _err = _run(project, monkeypatch, capsys, "install", "broadcast/tracks")
    assert code == 0
    assert "installed broadcast/tracks 1.0.0" in out

    entries = read_lockfile(project / "ffrwd.lock").entries
    assert len(entries) == 1
    entry = entries[0]
    assert isinstance(entry, RegistryEntry)
    assert (entry.name, entry.version) == ("broadcast/tracks", "1.0.0")
    assert entry.store == store.entry_path(entry.sha256)

    code, out, _err = _run(
        project, monkeypatch, capsys, "compile", QUERY.format(call="broadcast.tracks.quieter")
    )
    assert code == 0
    assert "volume=volume=0.5" in out


def test_install_records_the_dependency_keyed_by_name(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _publish(registry, _package(tmp_path / "built"))
    project = _project(tmp_path / "work", monkeypatch, capsys)
    before = _read_json(project / "ffrwd.json")

    code, out, _err = _run(project, monkeypatch, capsys, "install", "broadcast/tracks")
    assert code == 0
    assert "recorded in ffrwd.json as a dependency" in out

    after = _read_json(project / "ffrwd.json")
    assert after["dependencies"] == {"broadcast/tracks": "1.0.0"}
    # Everything the manifest already said is still there, unchanged.
    assert {key: after[key] for key in before} == before
    # And it still reads back through the same validation every command applies.
    assert read_manifest(project / "ffrwd.json").namespace == "consumer"


def test_the_lockfile_regenerates_byte_identically(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _publish(registry, _package(tmp_path / "built"))
    project = _project(tmp_path / "work", monkeypatch, capsys)
    assert _run(project, monkeypatch, capsys, "install", "broadcast/tracks")[0] == 0
    first = (project / "ffrwd.lock").read_bytes()
    assert _run(project, monkeypatch, capsys, "install", "broadcast/tracks")[0] == 0
    assert (project / "ffrwd.lock").read_bytes() == first
    assert _read_json(project / "ffrwd.json")["dependencies"] == {
        "broadcast/tracks": "1.0.0"
    }


def test_install_narrates_each_step_to_stderr(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _publish(registry, _package(tmp_path / "built"))
    project = _project(tmp_path / "work", monkeypatch, capsys)
    code, _out, err = _run(project, monkeypatch, capsys, "install", "broadcast/tracks")
    assert code == 0
    assert "resolving broadcast/tracks\n" in err
    assert "fetching broadcast/tracks 1.0.0 (" in err


def test_install_quiet_keeps_stderr_empty(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _publish(registry, _package(tmp_path / "built"))
    project = _project(tmp_path / "work", monkeypatch, capsys)
    code, out, err = _run(
        project, monkeypatch, capsys, "install", "-q", "broadcast/tracks"
    )
    assert code == 0
    assert err == ""
    assert "installed broadcast/tracks 1.0.0" in out


# ---------------------------------------------------------------------------
# install walks dependencies, recursively
# ---------------------------------------------------------------------------


def _dependent(root: Path, *, name: str, dep_name: str, dep_range: str) -> Path:
    """A package shipping one RECIPE that calls into `dep_name`'s default export."""
    (root / "queries").mkdir(parents=True, exist_ok=True)
    (root / "queries" / "run.sql").write_text(
        "-- variables: source (input media path), dest (output path)\n"
        f"COPY (SELECT {dep_name.replace('/', '.')}.quieter(f.audio[1]) "
        "FROM input(:'source') f) TO :'dest';\n",
        encoding="utf-8",
    )
    declared = {
        "name": name,
        "version": "1.0.0",
        "bin": {"thumb": "queries/run.sql"},
        "dependencies": {dep_name: dep_range},
    }
    (root / "ffrwd.json").write_text(json.dumps(declared, indent=2) + "\n", encoding="utf-8")
    return root


def test_install_fetches_a_dependency_and_its_recipe_runs_unaided(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The motivating case: a package with a dependency is not broken on arrival."""
    _publish(registry, _package(tmp_path / "video", name="broadcast/video", factor="0.5"))
    _publish(
        registry,
        _dependent(
            tmp_path / "images",
            name="broadcast/images",
            dep_name="broadcast/video",
            dep_range="^1.0.0",
        ),
        functions=(),
        recipes=("thumb",),
    )
    project = _project(tmp_path / "work", monkeypatch, capsys)

    code, out, _err = _run(project, monkeypatch, capsys, "install", "broadcast/images")
    assert code == 0
    assert "brought along as dependencies: broadcast/video 1.0.0" in out

    entries = read_lockfile(project / "ffrwd.lock").entries
    names = sorted(entry.name for entry in entries if isinstance(entry, RegistryEntry))
    assert names == ["broadcast/images", "broadcast/video"]
    # Only what was asked for is in the project's own manifest.
    assert _read_json(project / "ffrwd.json")["dependencies"] == {
        "broadcast/images": "1.0.0"
    }

    # Nothing installed by hand beyond `broadcast/images`, and its recipe runs.
    code, out, _err = _run(
        project,
        monkeypatch,
        capsys,
        "compile",
        "thumb",
        "-v",
        "source=in.mkv",
        "-v",
        "dest=out.mkv",
    )
    assert code == 0
    assert "volume=volume=0.5" in out


def test_a_dependency_already_pinned_at_the_wanted_version_is_not_brought_again(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _publish(registry, _package(tmp_path / "video", name="broadcast/video", factor="0.5"))
    _publish(
        registry,
        _dependent(
            tmp_path / "images",
            name="broadcast/images",
            dep_name="broadcast/video",
            dep_range="^1.0.0",
        ),
        functions=(),
        recipes=("thumb",),
    )
    project = _project(tmp_path / "work", monkeypatch, capsys)
    assert _run(project, monkeypatch, capsys, "install", "broadcast/video")[0] == 0

    installed = packages.install(
        "broadcast/images",
        lock=project / "ffrwd.lock",
        manifest=project / "ffrwd.json",
    )
    assert installed.brought == ()


def test_a_dependency_cycle_is_rejected_naming_the_loop(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _publish(
        registry,
        _package(tmp_path / "a", name="broadcast/a", dependencies={"broadcast/b": "^1.0.0"}),
    )
    _publish(
        registry,
        _package(tmp_path / "b", name="broadcast/b", dependencies={"broadcast/a": "^1.0.0"}),
    )
    project = _project(tmp_path / "work", monkeypatch, capsys)
    code, _out, err = _run(project, monkeypatch, capsys, "install", "broadcast/a")
    assert code == 1
    assert "dependency cycle" in err
    assert "broadcast/a" in err and "broadcast/b" in err
    assert read_lockfile(project / "ffrwd.lock").entries == ()


def test_two_installs_pin_different_versions_of_a_shared_dependency(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No resolver arbitrates: each dependent's own install keeps its own version."""
    _publish(registry, _package(tmp_path / "d1", name="broadcast/d", version="1.0.0"))
    _publish(
        registry,
        _package(tmp_path / "b", name="broadcast/b", dependencies={"broadcast/d": "^1.0.0"}),
    )
    project = _project(tmp_path / "work", monkeypatch, capsys)
    assert _run(project, monkeypatch, capsys, "install", "broadcast/b")[0] == 0

    _publish(registry, _package(tmp_path / "d2", name="broadcast/d", version="2.0.0"))
    _publish(
        registry,
        _package(tmp_path / "c", name="broadcast/c", dependencies={"broadcast/d": "^1.0.0"}),
    )
    assert _run(project, monkeypatch, capsys, "install", "broadcast/c")[0] == 0

    entries = read_lockfile(project / "ffrwd.lock").entries
    by_identity = {
        (entry.name, entry.version): entry for entry in entries if isinstance(entry, RegistryEntry)
    }
    assert sorted(v for (n, v) in by_identity if n == "broadcast/d") == ["1.0.0", "2.0.0"]
    assert by_identity[("broadcast/b", "1.0.0")].dependencies == {"broadcast/d": "1.0.0"}
    assert by_identity[("broadcast/c", "1.0.0")].dependencies == {"broadcast/d": "2.0.0"}


# ---------------------------------------------------------------------------
# versions
# ---------------------------------------------------------------------------


def test_no_version_installs_the_highest_published_one(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _publish(registry, _package(tmp_path / "v1_9", version="1.9.0", factor="0.9"))
    _publish(registry, _package(tmp_path / "v1_10", version="1.10.0", factor="0.10"))
    project = _project(tmp_path / "work", monkeypatch, capsys)

    code, out, _err = _run(project, monkeypatch, capsys, "install", "broadcast/tracks")
    assert code == 0
    # 1.10.0 over 1.9.0: dot-separated parts compared as numbers, not as text.
    assert "broadcast/tracks 1.10.0" in out
    assert read_lockfile(project / "ffrwd.lock").entries[0].version == "1.10.0"


def test_an_exact_version_is_taken_and_pinned(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _publish(registry, _package(tmp_path / "v1_9", version="1.9.0", factor="0.9"))
    _publish(registry, _package(tmp_path / "v1_10", version="1.10.0", factor="0.10"))
    project = _project(tmp_path / "work", monkeypatch, capsys)

    assert _run(project, monkeypatch, capsys, "install", "broadcast/tracks@1.9.0")[0] == 0
    assert read_lockfile(project / "ffrwd.lock").entries[0].version == "1.9.0"
    code, out, _err = _run(
        project, monkeypatch, capsys, "compile", QUERY.format(call="broadcast.tracks.quieter")
    )
    assert code == 0 and "volume=volume=0.9" in out


def test_a_version_the_registry_lacks_names_the_published_ones(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _publish(registry, _package(tmp_path / "built", version="1.0.0"))
    project = _project(tmp_path / "work", monkeypatch, capsys)
    code, _out, err = _run(project, monkeypatch, capsys, "install", "broadcast/tracks@2.0.0")
    assert code == 1
    assert "no version 2.0.0" in err and "published: 1.0.0" in err
    assert read_lockfile(project / "ffrwd.lock").entries == ()


def test_a_package_the_registry_lacks_is_refused(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _publish(registry, _package(tmp_path / "built"))
    project = _project(tmp_path / "work", monkeypatch, capsys)
    code, _out, err = _run(project, monkeypatch, capsys, "install", "someone/tracks")
    assert code == 1
    assert "no package 'someone/tracks'" in err
    assert "did you mean broadcast/tracks?" in err


def test_a_positional_that_is_not_a_package_name_is_refused(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _publish(registry, _package(tmp_path / "built"))
    project = _project(tmp_path / "work", monkeypatch, capsys)
    code, _out, err = _run(project, monkeypatch, capsys, "install", "../../etc/passwd")
    assert code == 1
    assert "does not name a package" in err


def test_a_reserved_namespace_is_refused_before_the_registry_is_asked(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _publish(registry, _package(tmp_path / "built"))
    project = _project(tmp_path / "work", monkeypatch, capsys)
    code, _out, err = _run(project, monkeypatch, capsys, "install", "ffmpeg/tracks")
    assert code == 1
    assert "namespace 'ffmpeg' is reserved" in err


# ---------------------------------------------------------------------------
# the store, and what a bad download costs
# ---------------------------------------------------------------------------


def test_a_tampered_archive_leaves_nothing_in_the_store_or_the_lockfile(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sha256 = _publish(registry, _package(tmp_path / "built"))
    swapped = store.pack(_package(tmp_path / "other", factor="9.9"))
    (registry / "archives" / sha256).write_bytes(swapped)
    # The size the registry records is the swapped archive's too, so what
    # rejects the download is the digest and nothing before it.
    detail = registry / "p" / "broadcast" / "tracks.json"
    payload = _read_json(detail)
    payload["versions"] = [
        {**dict(held), "size": len(swapped)} for held in list(payload["versions"])
    ]
    _write_json(detail, payload)
    project = _project(tmp_path / "work", monkeypatch, capsys)

    code, _out, err = _run(project, monkeypatch, capsys, "install", "broadcast/tracks")
    assert code == 1
    assert "was expected" in err and "nothing was written" in err
    assert "Traceback" not in err
    assert not (store.store_dir() / store.entry_path(sha256)).exists()
    assert read_lockfile(project / "ffrwd.lock").entries == ()
    assert "dependencies" not in _read_json(project / "ffrwd.json")


def test_an_archive_of_another_size_than_the_registry_recorded_is_refused(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sha256 = _publish(registry, _package(tmp_path / "built"))
    (registry / "archives" / sha256).write_bytes(b"")
    project = _project(tmp_path / "work", monkeypatch, capsys)
    code, _out, err = _run(project, monkeypatch, capsys, "install", "broadcast/tracks")
    assert code == 1
    assert "recorded" in err and "nothing was written" in err


def test_content_already_in_the_store_is_not_downloaded_again(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sha256 = _publish(registry, _package(tmp_path / "built"))
    first = _project(tmp_path / "one", monkeypatch, capsys)
    assert _run(first, monkeypatch, capsys, "install", "broadcast/tracks")[0] == 0

    # The only copy of the archive left is the one in the store.
    (registry / "archives" / sha256).unlink()
    second = _project(tmp_path / "two", monkeypatch, capsys)
    code, out, _err = _run(second, monkeypatch, capsys, "install", "broadcast/tracks")
    assert code == 0
    assert "already in the store" in out
    assert read_lockfile(second / "ffrwd.lock").entries[0].sha256 == sha256


# ---------------------------------------------------------------------------
# where an install may write
# ---------------------------------------------------------------------------


def test_install_outside_a_project_names_both_ways_forward(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _publish(registry, _package(tmp_path / "built"))
    bare = tmp_path / "elsewhere"
    bare.mkdir()
    code, _out, err = _run(bare, monkeypatch, capsys, "install", "broadcast/tracks")
    assert code == 2
    assert "ffrwd init" in err and "install -g" in err
    assert not (bare / "ffrwd.lock").exists()


def test_install_writes_the_machine_wide_lockfile(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _publish(registry, _package(tmp_path / "built"))
    bare = tmp_path / "elsewhere"
    bare.mkdir()
    code, out, _err = _run(bare, monkeypatch, capsys, "install", "-g", "broadcast/tracks")
    assert code == 0
    assert "recorded in" not in out  # there is no manifest beside a machine-wide lockfile
    entry = read_lockfile(store.global_lock_path()).entries[0]
    assert isinstance(entry, RegistryEntry) and entry.name == "broadcast/tracks"


def test_bare_install_fetches_the_manifests_dependencies_at_their_written_versions(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No package named: the manifest standing here is the request, pins and all."""
    monkeypatch.setenv(packages.REGISTRY_ENV, str(registry))
    _publish(registry, _package(tmp_path / "v1", version="1.0.0", factor="0.5"))
    _publish(registry, _package(tmp_path / "v2", version="2.0.0", factor="0.25"))
    project = _package(
        tmp_path / "work",
        name="consumer/mine",
        dependencies={"broadcast/tracks": "1.0.0"},
    )

    code, out, err = _run(project, monkeypatch, capsys, "install")
    assert code == 0, err
    assert "installed what consumer/mine 1.0.0 needs in" in out
    assert "fetched: broadcast/tracks 1.0.0" in out
    held = read_lockfile(project / "ffrwd.lock")
    assert held.dependencies == {"broadcast/tracks": "1.0.0"}
    versions = [
        entry.version for entry in held.entries if isinstance(entry, RegistryEntry)
    ]
    assert versions == ["1.0.0"]  # the written pin, not the highest published

    code, out, _err = _run(
        project, monkeypatch, capsys, "compile", QUERY.format(call="broadcast.tracks.quieter")
    )
    assert code == 0 and "volume=volume=0.5" in out

    code, out, _err = _run(project, monkeypatch, capsys, "install")
    assert code == 0
    assert "all dependencies were already pinned" in out


def test_bare_install_leaves_a_dependency_that_is_linked_to_a_directory(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A link is how a dependency nobody has published yet is worked against."""
    monkeypatch.setenv(packages.REGISTRY_ENV, str(registry))
    _package(tmp_path / "dev")  # the dependency, in a working tree and nowhere else
    project = _package(
        tmp_path / "work",
        name="consumer/mine",
        dependencies={"broadcast/tracks": "1.0.0"},
    )
    write_lockfile(project / "ffrwd.lock", [])
    assert _run(project, monkeypatch, capsys, "link", "../dev")[0] == 0

    code, out, err = _run(project, monkeypatch, capsys, "install")
    assert code == 0, err
    assert "broadcast/tracks is linked to a working directory" in err
    assert "installed what consumer/mine 1.0.0 needs in" in out
    held = read_lockfile(project / "ffrwd.lock")
    assert held.entries == (LinkEntry(path="../dev"),), "the link survived the install"
    assert held.dependencies == {}, "nothing was pinned to a published version"

    code, out, _err = _run(
        project, monkeypatch, capsys, "compile", QUERY.format(call="broadcast.tracks.quieter")
    )
    assert code == 0 and "volume=volume=0.5" in out


def test_bare_install_fetches_the_projects_own_pinned_models(
    store_home: Path,
    served: _Served,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The models a working tree pins land beside its own modules, once."""
    _serves(served, **{MODEL_URL: MODEL})
    project = _model_package(tmp_path / "work")

    code, _out, err = _run(project, monkeypatch, capsys, "install")
    assert code == 0, err
    assert (project / "depth.onnx").read_bytes() == MODEL
    assert served.urls() == [MODEL_URL]

    code, out, _err = _run(project, monkeypatch, capsys, "install")
    assert code == 0
    assert served.urls() == [MODEL_URL]  # already there and matching; not refetched
    assert "all dependencies were already pinned" in out


def test_bare_install_outside_a_manifest_names_both_ways_forward(
    store_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bare = tmp_path / "elsewhere"
    bare.mkdir()
    code, _out, err = _run(bare, monkeypatch, capsys, "install")
    assert code == 2
    assert "no ffrwd.json" in err and "ffrwd init" in err
    assert not (bare / "ffrwd.lock").exists()


def test_bare_install_with_g_needs_a_package_name(
    store_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bare = tmp_path / "elsewhere"
    bare.mkdir()
    code, _out, err = _run(bare, monkeypatch, capsys, "install", "-g")
    assert code == 2
    assert "-g needs a package name" in err


def test_installing_another_version_changes_the_want_but_keeps_the_old_entry(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """install never arbitrates between versions: the old one just stops being wanted."""
    _publish(registry, _package(tmp_path / "v1", version="1.0.0", factor="0.5"))
    _publish(registry, _package(tmp_path / "v2", version="2.0.0", factor="0.25"))
    project = _project(tmp_path / "work", monkeypatch, capsys)
    assert _run(project, monkeypatch, capsys, "install", "broadcast/tracks@1.0.0")[0] == 0

    code, out, _err = _run(project, monkeypatch, capsys, "install", "broadcast/tracks@2.0.0")
    assert code == 0
    assert "replacing the installed broadcast/tracks 1.0.0" in out
    entries = read_lockfile(project / "ffrwd.lock").entries
    versions = sorted(entry.version for entry in entries if isinstance(entry, RegistryEntry))
    assert versions == ["1.0.0", "2.0.0"]  # both still pinned; nothing is deleted
    assert _read_json(project / "ffrwd.json")["dependencies"] == {
        "broadcast/tracks": "2.0.0"
    }
    code, out, _err = _run(
        project, monkeypatch, capsys, "compile", QUERY.format(call="broadcast.tracks.quieter")
    )
    assert code == 0 and "volume=volume=0.25" in out  # the project resolves at its new want


def test_two_packages_under_one_namespace_install_and_both_stay_pinned(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No alias to collide over: two dependencies just sit in the lockfile by name."""
    _publish(registry, _package(tmp_path / "first", factor="0.5"))
    _publish(registry, _package(tmp_path / "second", name="broadcast/other", factor="0.25"))
    project = _project(tmp_path / "work", monkeypatch, capsys)
    assert _run(project, monkeypatch, capsys, "install", "broadcast/tracks")[0] == 0
    assert _run(project, monkeypatch, capsys, "install", "broadcast/other")[0] == 0

    listed = read_lockfile(project / "ffrwd.lock").entries
    assert sorted(entry.name for entry in listed if isinstance(entry, RegistryEntry)) == [
        "broadcast/other",
        "broadcast/tracks",
    ]
    assert _read_json(project / "ffrwd.json")["dependencies"] == {
        "broadcast/tracks": "1.0.0",
        "broadcast/other": "1.0.0",
    }
    code, out, _err = _run(
        project, monkeypatch, capsys, "compile", QUERY.format(call="broadcast.tracks.quieter")
    )
    assert code == 0 and "volume=volume=0.5" in out
    code, out, _err = _run(
        project, monkeypatch, capsys, "compile", QUERY.format(call="broadcast.other.quieter")
    )
    assert code == 0 and "volume=volume=0.25" in out


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def _catalogue(registry: Path, tmp_path: Path) -> None:
    """Two packages that share no field, so a match names which one answered."""
    _publish(
        registry,
        _package(
            tmp_path / "one",
            name="broadcast/tracks",
            description="audio track tools",
        ),
        functions=("quieter",),
    )
    _publish(
        registry,
        _package(
            tmp_path / "two",
            name="studio/captions",
            description="subtitle wrangling",
        ),
        functions=("burn_in",),
        recipes=("extract",),
    )


@pytest.mark.parametrize(
    ("term", "expected"),
    [
        ("tracks", "broadcast/tracks"),  # the package segment
        ("studio", "studio/captions"),  # the namespace segment
        ("subtitle", "studio/captions"),  # the description
        ("quieter", "broadcast/tracks"),  # a function it exports
        ("QUIETER", "broadcast/tracks"),  # case-insensitively
    ],
)
def test_search_filters_over_every_field_it_claims_to(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    term: str,
    expected: str,
) -> None:
    _catalogue(registry, tmp_path)
    code, out, _err = _run(tmp_path, monkeypatch, capsys, "search", term, "--json")
    assert code == 0
    assert [held["name"] for held in json.loads(out)["packages"]] == [expected]


def test_search_with_no_term_lists_everything(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _catalogue(registry, tmp_path)
    code, out, _err = _run(tmp_path, monkeypatch, capsys, "search")
    assert code == 0
    assert "broadcast/tracks" in out and "studio/captions" in out
    assert "(2 rows)" in out


def test_a_term_matching_nothing_is_an_empty_table_and_not_an_error(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _catalogue(registry, tmp_path)
    code, out, err = _run(tmp_path, monkeypatch, capsys, "search", "nothing-like-this")
    assert code == 0
    assert "(0 rows)" in out
    assert err == ""


def test_search_reads_no_project_and_works_in_a_bare_directory(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _catalogue(registry, tmp_path)
    bare = tmp_path / "elsewhere"
    bare.mkdir()
    assert _run(bare, monkeypatch, capsys, "search", "tracks")[0] == 0


# ---------------------------------------------------------------------------
# what the registry serves is data off the network
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("written", "needle"),
    [
        ("not json at all", "it is not JSON"),
        ("[]", "it is not a JSON object"),
        ('{"format_version": 99, "name": "broadcast/tracks", "versions": []}', "format version 99"),
        (
            '{"format_version": 2, "name": "broadcast/tracks", "versions": {}}',
            "'versions' is not a list of objects",
        ),
        ('{"format_version": 2, "versions": []}', "'name' is not a non-empty string"),
        (
            '{"format_version": 2, "name": "../etc", "versions": []}',
            "is not a package name",
        ),
        (
            '{"format_version": 2, "name": "ffmpeg/x", "versions": []}',
            "namespace 'ffmpeg' is reserved",
        ),
        (
            '{"format_version": 2, "name": "ffrwd/speed", "versions": []}',
            "is named for the macro ffrwd.speed()",
        ),
        (
            '{"format_version": 2, "name": "studio/captions", "versions": []}',
            "it describes another package than 'broadcast/tracks'",
        ),
    ],
)
def test_a_detail_document_this_client_cannot_read_is_refused(
    store_home: Path,
    registry: Path,
    monkeypatch: pytest.MonkeyPatch,
    written: str,
    needle: str,
) -> None:
    _write_json(registry / "p" / "broadcast" / "tracks.json", {})
    (registry / "p" / "broadcast" / "tracks.json").write_text(written, encoding="utf-8")
    with pytest.raises(FfrwdError) as caught:
        packages.resolve("broadcast/tracks")
    assert needle in caught.value.message, caught.value.message


def test_a_detail_file_with_a_bad_digest_is_refused(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _publish(registry, _package(tmp_path / "built"))
    detail = registry / "p" / "broadcast" / "tracks.json"
    payload = _read_json(detail)
    versions = list(payload["versions"])  # type: ignore[arg-type]
    versions[0] = {**dict(versions[0]), "sha256": "nope"}
    payload["versions"] = versions
    _write_json(detail, payload)

    project = _project(tmp_path / "work", monkeypatch, capsys)
    code, _out, err = _run(project, monkeypatch, capsys, "install", "broadcast/tracks")
    assert code == 1
    assert "is not a sha256 digest" in err


def test_an_archive_for_another_package_than_the_registry_listed_is_refused(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The digest matches, so the bytes are the ones published -- and the
    # package inside still says it is something else.
    _publish(registry, _package(tmp_path / "built", version="1.0.0"))
    detail = registry / "p" / "broadcast" / "tracks.json"
    payload = _read_json(detail)
    held_versions = list(payload["versions"])  # type: ignore[arg-type]
    versions = [{**dict(held), "version": "2.0.0"} for held in held_versions]
    payload["versions"] = versions
    _write_json(detail, payload)

    project = _project(tmp_path / "work", monkeypatch, capsys)
    code, _out, err = _run(project, monkeypatch, capsys, "install", "broadcast/tracks@2.0.0")
    assert code == 1
    assert "records version '2.0.0' and the package says '1.0.0'" in err
    assert read_lockfile(project / "ffrwd.lock").entries == ()


# ---------------------------------------------------------------------------
# the MCP tools
# ---------------------------------------------------------------------------


def test_the_search_tool_returns_what_matched_and_where_it_came_from(
    store_home: Path, registry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _catalogue(registry, tmp_path)
    result = mcp_tools.search_packages("subtitle")
    assert [held["name"] for held in result["packages"]] == ["studio/captions"]
    assert result["registry"] == str(registry)


def test_the_install_tool_installs_into_the_project_it_is_given(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _publish(registry, _package(tmp_path / "built"))
    project = _project(tmp_path / "work", monkeypatch, capsys)
    result = mcp_tools.install_package("broadcast/tracks", str(project))
    assert result["name"] == "broadcast/tracks"
    assert result["brought"] == []
    assert result["version"] == "1.0.0"
    assert result["downloaded"] is True
    assert read_lockfile(project / "ffrwd.lock").entries[0].name == "broadcast/tracks"
    assert capsys.readouterr().out == ""  # stdout is the protocol stream


def test_the_install_tool_refuses_a_directory_that_is_not_a_project(
    store_home: Path,
    registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _publish(registry, _package(tmp_path / "built"))
    bare = tmp_path / "elsewhere"
    bare.mkdir()
    with pytest.raises(FfrwdError) as caught:
        mcp_tools.install_package("broadcast/tracks", str(bare))
    assert "ffrwd.lock" in caught.value.message
    assert caught.value.hint is not None and "ffrwd init" in caught.value.hint


# ---------------------------------------------------------------------------
# the hosted registry: resolving over HTTP
# ---------------------------------------------------------------------------


def test_resolve_reads_the_detail_document_at_its_own_url(served: _Served) -> None:
    url = _detail_url("broadcast/tracks")
    # "readme_html" is a key this format version's parser has no use for: a
    # version object carries what it carries, and unknown keys are read past.
    version = {**_version("a" * 64, 11), "readme_html": "<h1>tracks</h1>\n"}
    _serves(served, **{url: _detail_document("broadcast/tracks", [version])})
    release = packages.resolve("broadcast/tracks")
    assert (release.version, release.sha256, release.size) == ("1.0.0", "a" * 64, 11)
    # One fetch, and no index step in front of it.
    assert served.urls() == [url]


def test_a_package_the_registry_lacks_suggests_what_search_ranks_first(
    served: _Served,
) -> None:
    _serves(
        served,
        **{
            SEARCH_URL: json.dumps(
                [
                    {
                        "name": "broadcast/tracks",
                        "version": "1.0.0",
                        "description": "audio track tools",
                    }
                ]
            ).encode("utf-8")
        },
    )
    with pytest.raises(FfrwdError) as caught:
        packages.resolve("someone/tracks")
    assert "the registry has no package 'someone/tracks'" in caught.value.message
    assert caught.value.hint == "did you mean broadcast/tracks?"


def test_a_search_that_cannot_be_reached_costs_the_suggestion_and_nothing_else(
    served: _Served,
) -> None:
    _serves(served, **{SEARCH_URL: (500, b"down")})
    with pytest.raises(FfrwdError) as caught:
        packages.resolve("someone/tracks")
    assert "the registry has no package 'someone/tracks'" in caught.value.message
    assert caught.value.hint == "run `ffrwd search` to see what is published"


def test_a_public_404_with_a_token_retries_the_private_endpoint(served: _Served) -> None:
    credentials.save(TOKEN, api=API)
    private = f"{API}/functions/v1/private/p/broadcast/tracks"
    _serves(
        served,
        **{private: _detail_document("broadcast/tracks", [_version("b" * 64, 12)])},
    )
    assert packages.resolve("broadcast/tracks").sha256 == "b" * 64
    assert served.urls() == [_detail_url("broadcast/tracks"), private]
    assert served.headers_for(private)["authorization"] == f"Bearer {TOKEN}"


def test_object_storages_400_for_a_missing_object_reads_as_absent(
    served: _Served,
) -> None:
    """Storage answers a missing PUBLIC object 400, and names the 404 in the body."""
    credentials.save(TOKEN, api=API)
    private = f"{API}/functions/v1/private/p/broadcast/tracks"
    _serves(
        served,
        **{
            _detail_url("broadcast/tracks"): (
                400,
                json.dumps(
                    {
                        "statusCode": "404",
                        "error": "not_found",
                        "message": "Object not found",
                        "code": "NoSuchKey",
                    }
                ).encode("utf-8"),
            ),
            private: _detail_document("broadcast/tracks", [_version("c" * 64, 13)]),
        },
    )
    assert packages.resolve("broadcast/tracks").sha256 == "c" * 64


def test_a_400_that_is_not_an_absent_object_stays_a_failure(served: _Served) -> None:
    _serves(served, **{_detail_url("broadcast/tracks"): (400, b'{"error": "bad range"}')})
    with pytest.raises(FfrwdError) as caught:
        packages.resolve("broadcast/tracks")
    assert "the registry could not be read at" in caught.value.message
    assert "HTTP 400" in caught.value.message


def test_a_private_detail_this_token_may_not_read_says_whose_fault_it_is(
    served: _Served,
) -> None:
    credentials.save(TOKEN, api=API)
    private = f"{API}/functions/v1/private/p/broadcast/tracks"
    _serves(served, **{private: (403, b"{}")})
    with pytest.raises(FfrwdError) as caught:
        packages.resolve("broadcast/tracks")
    assert "this token does not authorize reading 'broadcast/tracks'" in caught.value.message
    assert "mint one that is" in (caught.value.hint or "")


# ---------------------------------------------------------------------------
# the hosted registry: the search function
# ---------------------------------------------------------------------------


def test_search_posts_the_term_to_the_rpc_with_the_project_key(served: _Served) -> None:
    _serves(
        served,
        **{
            SEARCH_URL: json.dumps(
                [
                    {
                        "name": "broadcast/tracks",
                        "version": "1.2.0",
                        "description": "audio track tools",
                        "functions": ["quieter"],
                        "recipes": ["duck"],
                        "keywords": ["audio"],
                        "capabilities": ["nn"],
                        "score": 0.9,
                        "installs_week": 41,
                    }
                ]
            ).encode("utf-8")
        },
    )
    found = packages.search("ladder")
    assert json.loads(served.body_for(SEARCH_URL)) == {"q": "ladder"}
    headers = served.headers_for(SEARCH_URL)
    assert headers["apikey"] == packages.ANON_KEY
    assert headers["authorization"] == f"Bearer {packages.ANON_KEY}"
    assert len(found) == 1
    listing = found[0]
    assert (listing.name, listing.version, listing.installs_week) == (
        "broadcast/tracks",
        "1.2.0",
        41,
    )
    # Keys this client does not read are ignored rather than refused.
    assert listing.functions == ("quieter",) and listing.recipes == ("duck",)


def test_the_cli_search_table_shows_what_a_week_installed(
    served: _Served,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _serves(
        served,
        **{
            SEARCH_URL: json.dumps(
                [{"name": "broadcast/tracks", "version": "1.2.0", "installs_week": 41}]
            ).encode("utf-8")
        },
    )
    code, out, _err = _run(tmp_path, monkeypatch, capsys, "search", "tracks")
    assert code == 0
    assert "installs/week" in out and "41" in out


# ---------------------------------------------------------------------------
# the hosted registry: every archive comes through the signing endpoint
# ---------------------------------------------------------------------------


def _served_package(served: _Served, tmp_path: Path) -> tuple[str, bytes]:
    """Publish one package into a served registry: detail, sign answer, archive."""
    archive = store.pack(_package(tmp_path / "built"))
    sha256 = hashlib.sha256(archive).hexdigest()
    signed = f"https://blob.example/{sha256}?token=xyz"
    _serves(
        served,
        **{
            _detail_url("broadcast/tracks"): _detail_document(
                "broadcast/tracks", [_version(sha256, len(archive))]
            ),
            _sign_url(sha256): json.dumps(
                {"url": signed, "expires_at": "2026-08-28T00:05:00Z"}
            ).encode("utf-8"),
            signed: archive,
        },
    )
    return sha256, archive


def test_an_archive_is_asked_for_by_signing_and_then_downloaded(
    store_home: Path,
    served: _Served,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sha256, _archive = _served_package(served, tmp_path)
    project = _project(tmp_path / "work", monkeypatch, capsys)

    code, out, _err = _run(project, monkeypatch, capsys, "install", "broadcast/tracks")
    assert code == 0
    assert "installed broadcast/tracks 1.0.0" in out
    assert served.urls()[-2:] == [
        _sign_url(sha256),
        f"https://blob.example/{sha256}?token=xyz",
    ]
    # Nothing was logged in, so the sign request carried no bearer.
    assert "authorization" not in served.headers_for(_sign_url(sha256))
    assert read_lockfile(project / "ffrwd.lock").entries[0].sha256 == sha256


def test_a_version_that_is_not_public_with_no_token_says_to_log_in(
    store_home: Path, served: _Served, tmp_path: Path
) -> None:
    sha256, archive = _served_package(served, tmp_path)
    _serves(served, **{_sign_url(sha256): (401, b"{}")})
    release = packages.Release(
        name="broadcast/tracks", version="1.0.0", sha256=sha256, size=len(archive)
    )
    with pytest.raises(FfrwdError) as caught:
        packages.fetch(release)
    assert "holds no ffrwd token" in caught.value.message
    assert caught.value.hint == "run `ffrwd login --token <token>` and try again"


def test_a_version_this_token_may_not_download_says_whose_fault_it_is(
    store_home: Path, served: _Served, tmp_path: Path
) -> None:
    credentials.save(TOKEN, api=API)
    sha256, archive = _served_package(served, tmp_path)
    _serves(served, **{_sign_url(sha256): (403, b"{}")})
    release = packages.Release(
        name="broadcast/tracks", version="1.0.0", sha256=sha256, size=len(archive)
    )
    with pytest.raises(FfrwdError) as caught:
        packages.fetch(release)
    assert "this token does not authorize downloading" in caught.value.message
    assert served.headers_for(_sign_url(sha256))["authorization"] == f"Bearer {TOKEN}"


def test_a_signing_answer_that_names_no_url_is_refused(
    store_home: Path, served: _Served, tmp_path: Path
) -> None:
    sha256, archive = _served_package(served, tmp_path)
    _serves(served, **{_sign_url(sha256): b'{"expires_at": "soon"}'})
    release = packages.Release(
        name="broadcast/tracks", version="1.0.0", sha256=sha256, size=len(archive)
    )
    with pytest.raises(FfrwdError) as caught:
        packages.fetch(release)
    assert "'url' is not an http URL to download from" in caught.value.message


# ---------------------------------------------------------------------------
# the models a package pins, fetched from the hub
# ---------------------------------------------------------------------------

MODEL = b"not really an onnx graph, but it hashes to something"
MODEL_DIGEST = hashlib.sha256(MODEL).hexdigest()
MODEL_URL = "https://huggingface.co/depth-anything/small/resolve/v1/model.onnx"


def _model_package(root: Path, *, models: dict[str, object] | None = None) -> Path:
    """A package whose one export is a wasm module, pinning a model for it."""
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "lib.sql").write_text(
        "CREATE FUNCTION depth(v video_stream) RETURNS video_stream\n"
        "  AS 'depth.wasm', 'depth' LANGUAGE wasm;\n",
        encoding="utf-8",
    )
    (root / "depth.wasm").write_bytes(b"\0asm")
    declared: dict[str, object] = {
        "name": "broadcast/depth",
        "version": "1.0.0",
        "lib": {"depth": "src/lib.sql"},
        "models": models
        if models is not None
        else {
            "depth": {
                "repo": "depth-anything/small",
                "revision": "v1",
                "file": "model.onnx",
                "sha256": MODEL_DIGEST,
            }
        },
    }
    (root / "ffrwd.json").write_text(json.dumps(declared, indent=2) + "\n", encoding="utf-8")
    return root


def test_the_model_url_is_the_pin_written_out() -> None:
    pin = ModelPin(
        repo="depth-anything/small", revision="v1", file="model.onnx", sha256=MODEL_DIGEST
    )
    assert packages.model_url(pin) == MODEL_URL


def test_the_model_notice_carries_the_size_when_the_answer_names_one() -> None:
    pin = ModelPin(
        repo="depth-anything/small", revision="v1", file="model.onnx", sha256=MODEL_DIGEST
    )
    assert packages._model_notice(pin, 86_000_000) == (
        "model model.onnx (82 MB) from depth-anything/small"
    )
    assert packages._model_notice(pin, None) == "model model.onnx from depth-anything/small"
    answer = SimpleNamespace(headers={"Content-Length": "86000000"})
    assert packages._content_length(answer) == 86_000_000
    assert packages._content_length(SimpleNamespace()) is None


def test_a_pinned_model_lands_beside_the_module_hash_verified(
    store_home: Path,
    registry: Path,
    served: _Served,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A local registry serves the package; the model still comes from the hub."""
    monkeypatch.setenv(packages.REGISTRY_ENV, str(registry))
    sha256 = _publish(registry, _model_package(tmp_path / "built"), functions=("depth",))
    _serves(served, **{MODEL_URL: MODEL})
    project = _project(tmp_path / "work", monkeypatch, capsys)

    code, _out, err = _run(project, monkeypatch, capsys, "install", "broadcast/depth")
    assert code == 0, err
    landed = store.store_dir() / store.entry_path(sha256) / "depth.onnx"
    assert landed.read_bytes() == MODEL
    assert served.urls() == [MODEL_URL]
    # The fetch narrates the model; the fake answer names no Content-Length,
    # so the line carries no size.
    assert "model model.onnx from depth-anything/small\n" in err


def test_a_model_already_there_and_matching_is_not_fetched_again(
    store_home: Path,
    registry: Path,
    served: _Served,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(packages.REGISTRY_ENV, str(registry))
    _publish(registry, _model_package(tmp_path / "built"), functions=("depth",))
    _serves(served, **{MODEL_URL: MODEL})
    first = _project(tmp_path / "one", monkeypatch, capsys)
    assert _run(first, monkeypatch, capsys, "install", "broadcast/depth")[0] == 0

    second = _project(tmp_path / "two", monkeypatch, capsys)
    assert _run(second, monkeypatch, capsys, "install", "broadcast/depth")[0] == 0
    assert served.urls() == [MODEL_URL]  # once, for the first install


def test_a_model_that_hashes_to_something_else_fails_the_install(
    store_home: Path,
    registry: Path,
    served: _Served,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(packages.REGISTRY_ENV, str(registry))
    sha256 = _publish(registry, _model_package(tmp_path / "built"), functions=("depth",))
    _serves(served, **{MODEL_URL: b"something else entirely"})
    project = _project(tmp_path / "work", monkeypatch, capsys)

    code, _out, err = _run(project, monkeypatch, capsys, "install", "broadcast/depth")
    assert code == 1
    assert "the model for 'depth' (depth-anything/small@v1 model.onnx)" in err
    assert f"and {MODEL_DIGEST} was expected" in err
    assert not (store.store_dir() / store.entry_path(sha256) / "depth.onnx").exists()
    assert read_lockfile(project / "ffrwd.lock").entries == ()


def test_a_model_the_hub_does_not_serve_fails_the_install(
    store_home: Path,
    registry: Path,
    served: _Served,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(packages.REGISTRY_ENV, str(registry))
    _publish(registry, _model_package(tmp_path / "built"), functions=("depth",))
    project = _project(tmp_path / "work", monkeypatch, capsys)
    code, _out, err = _run(project, monkeypatch, capsys, "install", "broadcast/depth")
    assert code == 1
    assert "answered HTTP 404" in err
    assert read_lockfile(project / "ffrwd.lock").entries == ()


def test_a_model_naming_an_export_no_module_declares_fails_the_install(
    store_home: Path,
    registry: Path,
    served: _Served,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(packages.REGISTRY_ENV, str(registry))
    _publish(
        registry,
        _model_package(
            tmp_path / "built",
            models={
                "sharpen": {
                    "repo": "depth-anything/small",
                    "revision": "v1",
                    "file": "model.onnx",
                    "sha256": MODEL_DIGEST,
                }
            },
        ),
        functions=("depth",),
    )
    project = _project(tmp_path / "work", monkeypatch, capsys)
    code, _out, err = _run(project, monkeypatch, capsys, "install", "broadcast/depth")
    assert code == 1
    assert "the model for 'sharpen'" in err
    assert "names an export no module in it declares" in err
