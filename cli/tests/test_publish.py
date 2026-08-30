"""Tests for `ffrwd publish`: the local preflight, and the one request it makes.

The preflight is the whole validation, so most of what is checked here never
reaches the network at all: a manifest, its exports, its modules, its models
and every recipe compiled against the synthetic file. What does reach it goes
through ``packages._urlopen``, the same seam ``test_packages`` replaces --
here to resolve the dependencies the manifest names, and to make the upload.

No test invokes the sidecar: what a module declares comes from a
:class:`~ffrwd.wasm.Described` handed to the seam ``wasm.describe`` is, which
is the same shape a real one answers with.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
import urllib.error
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from ffrwd import cli, credentials, packages, publish, store, wasm
from ffrwd.console import written_size
from ffrwd.errors import FfrwdError
from ffrwd.project import read_manifest
from ffrwd.warnings import FfrwdWarning, WarningCode
from ffrwd.wasm import Described

API = "https://api.example"
STORAGE = "https://registry.example/packages"
TOKEN = "ffrwd_" + "d" * 43
PUBLISH_URL = f"{API}/functions/v1/publish"

RECIPE = (
    "-- Halve a file's audio and write it out.\n"
    "-- variables: source (input media path), dest (output path)\n"
    "-- example: ffrwd run broadcast.tracks.duck -v source=in.mkv -v dest=out.mkv\n"
    "COPY (SELECT broadcast.tracks.quieter(f.audio[1]) FROM input(:'source') f)"
    " TO :'dest';\n"
)


def _package(
    root: Path,
    *,
    name: str = "broadcast/tracks",
    version: str = "1.0.0",
    recipe: str = RECIPE,
    license: str | None = "MIT",
    extra: Mapping[str, object] | None = None,
) -> Path:
    """A package with one export and one recipe: the ordinary shape.

    `license` writes the manifest key; None leaves it out, which is what a
    publish warns about.
    """
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "recipes").mkdir(parents=True, exist_ok=True)
    (root / "src" / "lib.sql").write_text(
        "CREATE FUNCTION quieter(track audio_stream) RETURNS audio_stream AS $$\n"
        "  SELECT volume(track, 0.5)\n"
        "$$ LANGUAGE sql;\n",
        encoding="utf-8",
    )
    (root / "recipes" / "duck.sql").write_text(recipe, encoding="utf-8")
    declared: dict[str, object] = {
        "name": name,
        "version": version,
        "description": "audio track tools",
        "lib": {"quieter": "src/lib.sql"},
        "bin": {"duck": "recipes/duck.sql"},
        **({"license": license} if license is not None else {}),
        **(dict(extra) if extra else {}),
    }
    (root / "ffrwd.json").write_text(json.dumps(declared, indent=2) + "\n", encoding="utf-8")
    return root


def _module_package(
    root: Path,
    *,
    models: Mapping[str, object] | None = None,
    capabilities: list[str] | None = None,
) -> Path:
    """A package whose one export is a wasm module.

    `capabilities` writes the manifest key; None leaves it out, which is the
    same claim as an empty list.
    """
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
    }
    if capabilities is not None:
        declared["capabilities"] = list(capabilities)
    if models is not None:
        declared["models"] = dict(models)
    (root / "ffrwd.json").write_text(json.dumps(declared, indent=2) + "\n", encoding="utf-8")
    return root


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

    def sent_to(self, url: str) -> tuple[dict[str, str], bytes]:
        for asked, headers, body in self.asked:
            if asked == url:
                assert body is not None
                return headers, body
        raise AssertionError(f"nothing was sent to {url}")


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Served]:
    fake = _Served()
    monkeypatch.setenv(packages.REGISTRY_ENV, STORAGE)
    monkeypatch.setenv(packages.API_ENV, API)
    monkeypatch.setattr(packages, "_urlopen", fake)
    yield fake


def _accepts(served: _Served) -> None:
    served.answers[PUBLISH_URL] = json.dumps({"visibility": "public"}).encode("utf-8")


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
# the preflight
# ---------------------------------------------------------------------------


def test_the_preflight_describes_the_version_it_packed(tmp_path: Path) -> None:
    """The whole document, once: what the registry stores for this version."""
    root = _package(tmp_path / "built")
    prepared = publish.prepare(root / "ffrwd.json")

    archive = store.pack(root)
    assert prepared.archive == archive
    assert prepared.sha256 == hashlib.sha256(archive).hexdigest()
    # No modules and no "capabilities" key: absent is the same claim as empty,
    # so the declared and derived sets agree at nothing.
    assert prepared.package.capabilities == ()
    assert prepared.capabilities == ()
    assert prepared.sources == {"duck": RECIPE}
    assert prepared.detail == {
        "version": "1.0.0",
        "sha256": prepared.sha256,
        "size": len(archive),
        "namespace": "broadcast",
        "description": "audio track tools",
        "license": "MIT",
        "functions": [
            {
                "name": "quieter",
                "params": [{"name": "track", "type": "audio_stream"}],
                "returns": "audio_stream",
                "written": "quieter(track audio_stream)",
                "export": "src/lib.sql",
            }
        ],
        "recipes": [
            {
                "name": "duck",
                "file": "recipes/duck.sql",
                "description": "Halve a file's audio and write it out.",
                "usage": "ffrwd run broadcast.tracks.duck -v source=in.mkv -v dest=out.mkv",
                "required": [
                    {"name": "source", "description": "input media path"},
                    {"name": "dest", "description": "output path"},
                ],
                "optional": [],
            }
        ],
    }


def _members(archive: bytes) -> list[str]:
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as opened:
        return opened.getnames()


def test_a_readme_is_rendered_into_the_version_it_publishes(tmp_path: Path) -> None:
    """CommonMark to HTML, raw HTML through it: the site sanitizes, not this side."""
    root = _package(tmp_path / "built")
    (root / "README.md").write_text(
        '# duck\n\nHalve it. <span class="x">inline</span>\n', encoding="utf-8"
    )
    said: list[FfrwdWarning] = []
    prepared = publish.prepare(root / "ffrwd.json", on_warning=said.append)
    assert prepared.detail["readme_html"] == (
        '<h1>duck</h1>\n<p>Halve it. <span class="x">inline</span></p>\n'
    )
    assert said == []
    assert "README.md" in _members(prepared.archive)


def test_a_package_carrying_no_readme_is_warned_about_rather_than_refused(
    tmp_path: Path,
) -> None:
    root = _package(tmp_path / "built")
    said: list[FfrwdWarning] = []
    prepared = publish.prepare(root / "ffrwd.json", on_warning=said.append)
    assert "readme_html" not in prepared.detail
    assert [warning.code for warning in said] == [WarningCode.MISSING_README]
    assert "carries no README.md" in said[0].message


def test_the_license_a_manifest_declares_reaches_the_version_and_the_payload(
    tmp_path: Path,
) -> None:
    root = _package(tmp_path / "built", license="MIT OR Apache-2.0")
    said: list[FfrwdWarning] = []
    prepared = publish.prepare(root / "ffrwd.json", on_warning=said.append)
    assert prepared.detail["license"] == "MIT OR Apache-2.0"
    assert prepared.metadata()["license"] == "MIT OR Apache-2.0"
    assert WarningCode.MISSING_LICENSE not in [warning.code for warning in said]


def test_a_package_declaring_no_license_is_warned_about_rather_than_refused(
    tmp_path: Path,
) -> None:
    root = _package(tmp_path / "built", license=None)
    said: list[FfrwdWarning] = []
    prepared = publish.prepare(root / "ffrwd.json", on_warning=said.append)
    assert prepared.detail["license"] is None
    assert WarningCode.MISSING_LICENSE in [warning.code for warning in said]
    warned = next(one for one in said if one.code is WarningCode.MISSING_LICENSE)
    assert 'declares no "license"' in warned.message
    assert warned.hint is not None and "license" in warned.hint


def test_a_recipe_that_does_not_compile_is_refused_naming_it(tmp_path: Path) -> None:
    root = _package(
        tmp_path / "built",
        recipe="-- variables: source (input media path)\n"
        "COPY (SELECT nosuchfilter(f.audio[1]) FROM input(:'source') f) TO 'out.mkv';\n",
    )
    with pytest.raises(FfrwdError) as caught:
        publish.prepare(root / "ffrwd.json")
    assert "recipe 'duck' does not compile" in caught.value.message
    assert "publish again" in (caught.value.hint or "")


def test_an_export_the_lib_file_does_not_define_is_refused(tmp_path: Path) -> None:
    root = _package(tmp_path / "built")
    (root / "ffrwd.json").write_text(
        json.dumps(
            {
                "name": "broadcast/tracks",
                "version": "1.0.0",
                "lib": {"louder": "src/lib.sql"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with pytest.raises(FfrwdError) as caught:
        publish.prepare(root / "ffrwd.json")
    assert "does not define 'louder'" in caught.value.message


def test_a_package_named_for_a_macro_is_refused_here_too(tmp_path: Path) -> None:
    """The compiler's macro list is what publishing checks against."""
    root = _package(tmp_path / "built", name="ffrwd/speed")
    with pytest.raises(FfrwdError) as caught:
        publish.prepare(root / "ffrwd.json")
    assert "is named for the macro ffrwd.speed()" in caught.value.message


def test_a_dependency_the_registry_cannot_resolve_is_refused(
    served: _Served, tmp_path: Path
) -> None:
    root = _package(tmp_path / "built", extra={"dependencies": {"studio/captions": "^1.0.0"}})
    with pytest.raises(FfrwdError) as caught:
        publish.prepare(root / "ffrwd.json")
    assert "depends on 'studio/captions', which the registry could not resolve" in (
        caught.value.message
    )
    assert "publish what it depends on first" in (caught.value.hint or "")


def _describing(
    monkeypatch: pytest.MonkeyPatch, *, nn: bool, http: bool = False, udp: bool = False
) -> None:
    """The sidecar seam, answering for every module with the same description."""
    monkeypatch.setattr(
        wasm,
        "describe",
        lambda path: Described(
            world="ffrwd:av", name="depth", nn=nn, http=http, udp=udp
        ),
    )


def test_what_the_modules_declare_becomes_the_capabilities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The declared set and the derived set agree, and the derived one is published."""
    root = _module_package(tmp_path / "built", capabilities=["nn"])
    _describing(monkeypatch, nn=True)
    prepared = publish.prepare(root / "ffrwd.json")
    assert prepared.capabilities == ("nn",)
    assert prepared.package.capabilities == ("nn",)


def test_every_effect_a_module_imports_becomes_a_capability(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """http and udp derive from the describe the same way nn does, name-sorted."""
    root = _module_package(tmp_path / "built", capabilities=["http", "nn", "udp"])
    _describing(monkeypatch, nn=True, http=True, udp=True)
    prepared = publish.prepare(root / "ffrwd.json")
    assert prepared.capabilities == ("http", "nn", "udp")


def test_a_module_needing_a_capability_the_manifest_omits_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _module_package(tmp_path / "built")
    _describing(monkeypatch, nn=True)
    with pytest.raises(FfrwdError) as caught:
        publish.prepare(root / "ffrwd.json")
    assert (
        "the module 'depth.wasm' needs the 'nn' capability, which the manifest does not declare"
        in caught.value.message
    )
    assert '"capabilities": ["nn"]' in (caught.value.hint or "")


def test_a_declared_capability_no_module_needs_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _module_package(tmp_path / "built", capabilities=["nn"])
    _describing(monkeypatch, nn=False)
    with pytest.raises(FfrwdError) as caught:
        publish.prepare(root / "ffrwd.json")
    assert "the manifest declares the 'nn' capability" in caught.value.message
    assert "no module in this package needs it" in (caught.value.hint or "")


def test_a_model_naming_no_export_of_this_package_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _module_package(
        tmp_path / "built",
        models={
            "sharpen": {
                "repo": "depth-anything/small",
                "revision": "v1",
                "file": "model.onnx",
                "sha256": "e" * 64,
            }
        },
    )
    monkeypatch.setattr(
        wasm, "describe", lambda path: Described(world="ffrwd:av", name="depth")
    )
    with pytest.raises(FfrwdError) as caught:
        publish.prepare(root / "ffrwd.json")
    assert "the model for 'sharpen' names an export no module in this package declares" in (
        caught.value.message
    )
    assert "this package declares depth" in (caught.value.hint or "")


def test_an_archive_over_the_cap_says_where_a_model_belongs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _package(tmp_path / "built")
    monkeypatch.setattr(publish, "_MAX_ARCHIVE_BYTES", 8)
    with pytest.raises(FfrwdError) as caught:
        publish.prepare(root / "ffrwd.json")
    assert "and at most 8 are published" in caught.value.message
    assert "install fetches it from the hub" in (caught.value.hint or "")


def test_the_preflight_puts_the_compiler_probe_back_when_it_is_done(tmp_path: Path) -> None:
    """A publish runs in the same process as whatever asked for it."""
    from ffrwd import compiler

    held = compiler.probe_path
    publish.prepare(_package(tmp_path / "built") / "ffrwd.json")
    assert compiler.probe_path is held


# ---------------------------------------------------------------------------
# the upload
# ---------------------------------------------------------------------------


def test_publishing_without_a_token_says_to_log_in(served: _Served, tmp_path: Path) -> None:
    prepared = publish.prepare(_package(tmp_path / "built") / "ffrwd.json")
    with pytest.raises(FfrwdError) as caught:
        publish.publish(prepared)
    assert "publishing needs an ffrwd token" in caught.value.message
    assert caught.value.hint == "run `ffrwd login --token <token>` first"
    assert served.asked == []


def test_the_upload_carries_the_archive_and_the_metadata_in_one_request(
    served: _Served, tmp_path: Path
) -> None:
    credentials.save(TOKEN, api=API)
    _accepts(served)
    root = _package(
        tmp_path / "built",
        extra={"keywords": ["audio", "loudness"], "ffrwd": ">=0.9"},
    )
    prepared = publish.prepare(root / "ffrwd.json")
    published = publish.publish(prepared)

    assert [url for url, _headers, _body in served.asked] == [PUBLISH_URL]
    headers, body = served.sent_to(PUBLISH_URL)
    assert headers["authorization"] == f"Bearer {TOKEN}"
    assert headers["content-type"].startswith("multipart/form-data; boundary=")
    assert prepared.archive in body
    assert b'name="archive"; filename="broadcast-tracks-1.0.0.tgz"' in body
    assert b'name="metadata"; filename="metadata.json.gz"' in body

    metadata = json.loads(
        gzip.decompress(
            body.split(b"\r\n\r\n", 1)[1].split(b"\r\n--", 1)[0]
        ).decode("utf-8")
    )
    assert set(metadata) == {
        "name",
        "version",
        "sha256",
        "visibility",
        "detail",
        "sources",
        "capabilities",
        "engines",
        "keywords",
        "license",
        "models",
    }
    assert metadata["name"] == "broadcast/tracks"
    assert metadata["version"] == "1.0.0"
    assert metadata["sha256"] == prepared.sha256
    assert metadata["detail"] == prepared.detail
    assert metadata["sources"] == {"duck": RECIPE}
    assert metadata["capabilities"] == []
    assert metadata["engines"] == ">=0.9"
    assert metadata["keywords"] == ["audio", "loudness"]
    assert metadata["models"] == {}
    assert metadata["visibility"] == "public"
    assert published.visibility == "public"


def test_private_rides_the_payload_and_the_registry_answers_with_the_stamp(
    served: _Served,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The manifest's own "private" key is what stamps the version."""
    credentials.save(TOKEN, api=API)
    served.answers[PUBLISH_URL] = json.dumps({"visibility": "private"}).encode("utf-8")
    root = _package(tmp_path / "built")
    manifest_path = root / "ffrwd.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["private"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    code, out, err = _run(root, monkeypatch, capsys, "publish")
    assert code == 0, err

    _headers, body = served.sent_to(PUBLISH_URL)
    metadata = json.loads(
        gzip.decompress(
            body.split(b"\r\n\r\n", 1)[1].split(b"\r\n--", 1)[0]
        ).decode("utf-8")
    )
    assert metadata["visibility"] == "private"
    assert "published broadcast/tracks 1.0.0 (private)" in out


def test_the_registrys_own_refusal_is_what_the_user_reads(
    served: _Served, tmp_path: Path
) -> None:
    credentials.save(TOKEN, api=API)
    refused = (
        "package 'broadcast/tracks': version 1.0.0 is published with archive "
        f"{'a' * 64} and this source packs to {'b' * 64}"
    )
    served.answers[PUBLISH_URL] = (
        409,
        json.dumps(
            {
                "error": refused,
                "hint": "a published version may not change: bump the version in "
                "its manifest",
            }
        ).encode("utf-8"),
    )
    prepared = publish.prepare(_package(tmp_path / "built") / "ffrwd.json")
    with pytest.raises(FfrwdError) as caught:
        publish.publish(prepared)
    assert caught.value.message == refused  # verbatim, not paraphrased
    assert caught.value.hint == (
        "a published version may not change: bump the version in its manifest"
    )


def test_a_refusal_that_says_nothing_still_names_its_status(
    served: _Served, tmp_path: Path
) -> None:
    credentials.save(TOKEN, api=API)
    served.answers[PUBLISH_URL] = (503, b"<html>down</html>")
    prepared = publish.prepare(_package(tmp_path / "built") / "ffrwd.json")
    with pytest.raises(FfrwdError) as caught:
        publish.publish(prepared)
    assert "refused the publish with HTTP 503" in caught.value.message


# ---------------------------------------------------------------------------
# the command
# ---------------------------------------------------------------------------


def test_publish_validates_then_uploads_and_says_both(
    served: _Served,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    credentials.save(TOKEN, api=API)
    _accepts(served)
    root = _package(tmp_path / "built")
    code, out, err = _run(root, monkeypatch, capsys, "publish")
    assert code == 0, err
    assert "validated broadcast/tracks 1.0.0" in out
    assert "published broadcast/tracks 1.0.0 (public)" in out
    assert "ffrwd install broadcast/tracks" in out
    assert read_manifest(root / "ffrwd.json").version == "1.0.0"  # nothing was rewritten


def test_prepare_and_publish_announce_their_steps(served: _Served, tmp_path: Path) -> None:
    credentials.save(TOKEN, api=API)
    _accepts(served)
    root = _package(tmp_path / "built")
    said: list[str] = []
    prepared = publish.prepare(root / "ffrwd.json", announce=said.append)
    publish.publish(prepared, announce=said.append)
    assert said == [
        "validating broadcast/tracks 1.0.0",
        f"packing {root}",
        f"uploading broadcast/tracks 1.0.0 ({written_size(prepared.size)})",
    ]


def test_publish_narrates_to_stderr_and_quiet_silences_it(
    served: _Served,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    credentials.save(TOKEN, api=API)
    _accepts(served)
    root = _package(tmp_path / "built")
    code, out, err = _run(root, monkeypatch, capsys, "publish")
    assert code == 0
    assert "validating broadcast/tracks 1.0.0\n" in err
    assert "uploading broadcast/tracks 1.0.0 (" in err

    code, out, err = _run(root, monkeypatch, capsys, "publish", "-q")
    assert code == 0
    assert "published broadcast/tracks 1.0.0 (public)" in out
    # Quiet mutes the narration, not the warnings: a missing README still says so.
    assert "validating" not in err and "uploading" not in err
    assert "carries no README.md" in err


def test_publish_refuses_before_uploading_anything(
    served: _Served,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    credentials.save(TOKEN, api=API)
    _accepts(served)
    root = _package(
        tmp_path / "built",
        recipe="-- variables: source (input media path)\n"
        "COPY (SELECT nosuchfilter(f.audio[1]) FROM input(:'source') f) TO 'out.mkv';\n",
    )
    code, out, err = _run(root, monkeypatch, capsys, "publish")
    assert code == 1
    assert out == ""
    assert "recipe 'duck' does not compile" in err
    assert served.asked == []
