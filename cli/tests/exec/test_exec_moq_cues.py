"""A live program read back with ffrwd/moq's subscribe, its captions with it.

Marked ``@pytest.mark.exec`` and excluded from the default run. It needs a
MoQ relay and the ffrwd/moq package, so it runs only where both are named::

    FFRWD_MOQ=<package> MOQ_RELAY=<moq-relay> \\
        python -m pytest -m exec tests/exec/test_exec_moq_cues.py -q

``FFRWD_MOQ`` is an ffrwd/moq 0.12.1 package directory, one ``ffrwd install
ffrwd/moq@0.12.1`` put in the store or a checkout with its modules built;
the test links it into a HOME of its own. ``MOQ_RELAY`` is a moq-relay
binary, else ``moq-relay`` on PATH.

The program is published as a broadcast is, by ``ffrwd run`` of a query
into ``ffrwd.moq.publish`` over a relay on the loopback: a 60 fps H.264
picture at a constant bitrate, its sound, and a text track of the cue rows
the sidecar fleet's ``pauses`` module writes, an utterance every 14 s. The
publisher holds a text track three seconds behind the newest picture it has
seen, so subscribe's text port says how far it has got three seconds behind
the picture beside it. ffmpeg reads no input more than a tenth of a second
ahead of an output's furthest-behind stream: with the cue stream's
heartbeats on the text port's time, three seconds of a 6 Mbit/s picture
would have to wait between the sidecar and ffmpeg, and the run would end
wedged, reported as an overflow. The cue stream beats at the picture's time.

Requires ``ffmpeg``/``ffprobe`` on PATH with libx264, the ``ffrwd-wasm``
sidecar, the ``pauses`` module built for ``wasm32-wasip2``, and Python's
``cryptography`` for the relay's certificate. Tests skip cleanly when any of
those is missing.
"""

from __future__ import annotations

import datetime
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ffrwd import binaries

pytestmark = pytest.mark.exec

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PAUSES = (
    _REPO_ROOT / "sidecar" / "modules" / "target" / "wasm32-wasip2" / "release" / "pauses.wasm"
)
_BROADCAST = "jobs/test/out"
_SECONDS = 52
_RATE = 60
_EVERY = 14
# The utterances the program says whose rows reach the end of it, by start:
# the first is said before a reader joins. Each row is whole half a second
# after its utterance ends, well within the three seconds the publisher holds
# its text track back, so each cue keeps its own start.
_SAID = {14.25: "Bob", 28.25: "Ann", 42.25: "Bob"}
_TIMING = re.compile(r"((?:\d+:)?\d+:\d+\.\d+) --> ((?:\d+:)?\d+:\d+\.\d+)")

_PUBLISH = """
CREATE FUNCTION pauses(v video_stream, every number DEFAULT 12, lasts number DEFAULT 2.5,
                       late number DEFAULT 1)
  RETURNS cue[] AS :'pauses', 'pauses' LANGUAGE wasm;
COPY (
  SELECT f.video[1], f.audio[1],
         pauses(f.video[1], every => :every, lasts => 2, late => 0.5) AS captions
  FROM input(:'program', realtime => true) f
) TO ffrwd.moq.publish(:'relay', :'broadcast', :'cert')
"""

# The cluster's HLS egress piece, as it runs there.
_HLS = """
COPY (SELECT s.video, s.audio , s.data[1] AS captions
  FROM ffrwd.moq.subscribe(:'subscribe_relay', :'subscribe_broadcast', :'subscribe_cert',
                           reconnect_s => 0) s
) TO :'master' WITH (format 'hls', hls_time 4, hls_segment_type 'fmp4', hls_list_size 150,
  hls_start_number_source 'epoch', hls_fmp4_init_filename :'init',
  hls_flags 'delete_segments+program_date_time+independent_segments+temp_file')
"""

_MKV = """
COPY (SELECT s.video, s.audio, s.data[1] AS captions
  FROM ffrwd.moq.subscribe(:'subscribe_relay', :'subscribe_broadcast', :'subscribe_cert',
                           reconnect_s => 0) s
) TO :'master'
"""


def _relay_binary() -> str | None:
    named = os.environ.get("MOQ_RELAY")
    return named if named else shutil.which("moq-relay")


def _package() -> Path | None:
    named = os.environ.get("FFRWD_MOQ")
    if not named or not (Path(named) / "ffrwd.json").is_file():
        return None
    return Path(named)


@pytest.fixture(autouse=True)
def _require_tools() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found on PATH")
    if binaries.ffrwd_wasm_path() is None:
        pytest.skip("ffrwd-wasm not found")
    if not _PAUSES.exists():
        pytest.skip(f"module missing: {_PAUSES} (cargo build --target wasm32-wasip2 --release)")
    if _package() is None:
        pytest.skip("FFRWD_MOQ names no ffrwd/moq package")
    if _relay_binary() is None:
        pytest.skip("no moq-relay: set MOQ_RELAY")
    pytest.importorskip("cryptography")


def _certificate(into: Path) -> str:
    """A self-signed certificate for the loopback, written as the relay reads
    it; its DER as hex, which publish and subscribe trust outright."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=10))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    (into / "cert.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    (into / "key.pem").write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return cert.public_bytes(serialization.Encoding.DER).hex()


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as held:
        held.bind(("127.0.0.1", 0))
        return int(held.getsockname()[1])


def _make_program(path: Path, size: str, bitrate: str) -> None:
    args = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={_RATE}",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
        "-t", str(_SECONDS), "-c:v", "libx264", "-preset", "veryfast", "-bf", "2",
        "-b:v", bitrate, "-minrate", bitrate, "-maxrate", bitrate, "-bufsize", bitrate,
        "-x264-params", "nal-hrd=cbr", "-g", str(2 * _RATE), "-pix_fmt", "yuv420p",
        "-ac", "2", "-c:a", "aac", "-b:a", "128k", "-f", "nut", str(path),
    ]  # fmt: skip
    made = subprocess.run(args, capture_output=True, text=True, timeout=900)
    assert made.returncode == 0, made.stderr[-1500:]


@pytest.fixture(scope="module")
def programs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The program at 1080p and 6 Mbit/s, which fills any pipe within the
    text track's three seconds, and at 180p and 120 kbit/s, which does not."""
    into = tmp_path_factory.mktemp("programs")
    made = {}
    for name, size, bitrate in (("1080p", "1920x1080", "6000k"), ("180p", "320x180", "120k")):
        made[name] = into / f"{name}.nut"
        _make_program(made[name], size, bitrate)
    return made


def _env(home: Path) -> dict[str, str]:
    """A HOME of the test's own, the package linked into it, for a CLI run."""
    cache = home / ".cache" / "ffrwd"
    cache.mkdir(parents=True, exist_ok=True)
    links = {
        "format_version": 2,
        "machine_local": "the test's own",
        "links": [{"name": "ffrwd/moq", "path": str(_package())}],
    }
    (cache / "ffrwd.links").write_text(json.dumps(links), encoding="utf-8")
    env = dict(os.environ, HOME=str(home), USERPROFILE=str(home))
    env["FFRWD_WASM"] = str(binaries.ffrwd_wasm_path())
    return env


def _ffrwd(*args: str) -> list[str]:
    return [sys.executable, "-m", "ffrwd", *args]


def _spawn(args: list[str], **kwargs: object) -> subprocess.Popen[str]:
    """A child in a process group of its own, so a stop reaches what it
    started as well."""
    if os.name != "nt":
        kwargs.setdefault("start_new_session", True)
    return subprocess.Popen(args, text=True, **kwargs)  # type: ignore[call-overload]


def _stop(child: subprocess.Popen[str]) -> None:
    """Stops `child` and everything it started, where it has not ended."""
    if child.poll() is None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(child.pid)],
                           capture_output=True, timeout=30)  # fmt: skip
        else:
            import signal

            try:
                os.killpg(os.getpgid(child.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    child.wait(timeout=30)


def _broadcast_up(params: dict[str, str], env: dict[str, str]) -> bool:
    """Whether subscribe sees the broadcast: its shape is the catalog read."""
    subscribe = _package_module("subscribe")
    args = [
        env["FFRWD_WASM"], "--shape", subscribe, "-udp", subscribe, "-http", subscribe,
        "-params", json.dumps(params),
    ]  # fmt: skip
    return subprocess.run(args, capture_output=True, timeout=120, env=env).returncode == 0


def _package_module(name: str) -> str:
    package = _package()
    assert package is not None
    return str(package / "target" / "wasm32-wasip2" / "release" / f"{name}.wasm")


def _run_piece(
    tmp_path: Path, program: Path, piece: str, master: Path, *variables: str
) -> None:
    """Publishes `program` and runs `piece` over it to its end: the
    publisher's tracks finish with the program, and subscribe with them."""
    certs = tmp_path / "certs"
    certs.mkdir()
    cert = _certificate(certs)
    port = _free_udp_port()
    relay_url = f"moqt://127.0.0.1:{port}"
    env = _env(tmp_path / "home")
    relay = _spawn(
        [
            str(_relay_binary()), "--server-bind", f"127.0.0.1:{port}",
            "--tls-cert", str(certs / "cert.pem"), "--tls-key", str(certs / "key.pem"),
            "--auth-public", "",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    publisher = run = None
    try:
        (tmp_path / "publish.sql").write_text(_PUBLISH, encoding="utf-8")
        published = tmp_path / "publisher.err"
        with open(published, "w", encoding="utf-8") as said:
            publisher = _spawn(
                _ffrwd(
                    "run", "-q", "-f", str(tmp_path / "publish.sql"), "-y",
                    "-v", f"pauses={_PAUSES.as_posix()}", "-v", f"every={_EVERY}",
                    "-v", f"program={program}", "-v", f"relay={relay_url}",
                    "-v", f"broadcast={_BROADCAST}", "-v", f"cert={cert}",
                ),
                stdout=subprocess.DEVNULL, stderr=said, env=env, cwd=tmp_path,
            )  # fmt: skip
        params = {"relay": relay_url, "broadcast": _BROADCAST, "cert": cert}
        until = time.monotonic() + 120
        while not _broadcast_up(params, env):
            assert publisher.poll() is None, published.read_text(encoding="utf-8")[-3000:]
            assert time.monotonic() < until, "the broadcast never came up"
            time.sleep(0.5)
        (tmp_path / "piece.sql").write_text(piece, encoding="utf-8")
        named = [word for variable in variables for word in ("-v", variable)]
        said = [
            "-f", str(tmp_path / "piece.sql"),
            "-v", f"subscribe_relay={relay_url}", "-v", f"subscribe_broadcast={_BROADCAST}",
            "-v", f"subscribe_cert={cert}", "-v", f"master={master}", *named,
        ]  # fmt: skip
        # subscribe never ends, so the run is live: its muxer holds the
        # picture back for a cue a tenth of a second at most.
        compiled = subprocess.run(
            _ffrwd("compile", "-q", *said), capture_output=True, text=True, env=env,
            cwd=tmp_path, timeout=120,
        )  # fmt: skip
        assert compiled.returncode == 0, compiled.stderr[-3000:]
        assert "-max_interleave_delta 100000" in compiled.stdout, compiled.stdout
        run = _spawn(
            _ffrwd("run", "-q", "-y", *said),
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env, cwd=tmp_path,
        )  # fmt: skip
        _, stderr = run.communicate(timeout=_SECONDS + 120)
        assert "BUFFER_OVERFLOW" not in stderr, stderr[-3000:]
        assert run.returncode == 0, stderr[-3000:]
    finally:
        for child in (run, publisher, relay):
            if child is not None:
                _stop(child)


def _seconds(stamp: str) -> float:
    parts = [float(part) for part in stamp.split(":")]
    return sum(part * 60**power for power, part in enumerate(reversed(parts)))


def _cues(webvtt: str) -> list[tuple[float, float, str]]:
    """Each cue of a WebVTT document: its start, its end and its text."""
    cues: list[tuple[float, float, str]] = []
    for block in webvtt.split("\n\n"):
        lines = block.strip().splitlines()
        timing = next((line for line in lines if _TIMING.search(line)), None)
        if timing is None:
            continue
        found = _TIMING.search(timing)
        assert found is not None
        text = "\n".join(lines[lines.index(timing) + 1 :])
        cues.append((_seconds(found[1]), _seconds(found[2]), text))
    return cues


def _check_said(cues: list[tuple[float, float, str]]) -> None:
    """Every utterance said after the reader joined, at its own time on the
    program's timeline, said by the one who said it, and nothing else."""
    starts = {round(start, 2): text for start, end, text in cues if end > start}
    assert len(cues) == len(starts), cues
    said = {start: voice for start, voice in _SAID.items() if start in starts}
    assert said == _SAID, cues
    for start, voice in _SAID.items():
        assert starts[start] == f"<v {voice}>cue {int(start // _EVERY)} said at {start:.2f}"


@pytest.mark.parametrize("program_name", ["1080p", "180p"])
def test_a_live_programs_captions_reach_hls_as_the_cluster_reads_them(
    tmp_path: Path, programs: dict[str, Path], program_name: str
) -> None:
    hls = tmp_path / "hls"
    hls.mkdir()
    _run_piece(tmp_path, programs[program_name], _HLS, hls / "master.m3u8", "init=init.mp4")
    variant = next(path for path in hls.iterdir() if path.is_dir())
    assert len(list(variant.glob("segment_*.m4s"))) >= 10
    cues: list[tuple[float, float, str]] = []
    for segment in sorted(variant.glob("index*.vtt")):
        text = segment.read_text(encoding="utf-8")
        assert text.startswith("WEBVTT\nX-TIMESTAMP-MAP=MPEGTS:0,LOCAL:00:00:00.000\n")
        cues += _cues(text)
    _check_said(cues)


def test_a_live_programs_captions_reach_a_file_as_a_subtitle_track(
    tmp_path: Path, programs: dict[str, Path]
) -> None:
    out = tmp_path / "out.mkv"
    _run_piece(tmp_path, programs["1080p"], _MKV, out)
    extracted = subprocess.run(
        ["ffmpeg", "-v", "error", "-copyts", "-i", str(out), "-map", "0:s:0",
         "-c:s", "copy", "-f", "webvtt", "-"],
        capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    assert extracted.returncode == 0, extracted.stderr
    _check_said(_cues(extracted.stdout))
