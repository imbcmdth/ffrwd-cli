"""Places the pinned libav libraries for one wheel target in ffrwd_wasm_av/.

    python fetch_av.py --target <maturin target> [--os <runner os>]
    python fetch_av.py --check

The wheel ships ffrwd_wasm_av/ at its root. The archive for the target's
platform is downloaded, checked against the sha256 and byte count in
av-pins.json, and its four shared libraries, COPYING.LGPLv2.1, VERSION and
CONFIGURE replace whatever the directory held. A platform with no pin
leaves the directory holding only a README, and the wheel builds without
codecs. --check only verifies av-pins.json's shape.

Standard library only: this runs before anything is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
PINS = HERE / "av-pins.json"
DEST = HERE / "ffrwd_wasm_av"

LIBRARIES = ("avutil", "swresample", "avcodec", "swscale")
EXTRAS = ("COPYING.LGPLv2.1", "VERSION", "CONFIGURE")
# The MIT notices of the header sets a platform compiles in, present only
# where it does.
NOTICES = ("LICENSE.nv-codec-headers", "LICENSE.AMF")

# (runner OS, maturin target) -> platform, spelled as the builds' archives are.
PLATFORMS = {
    ("linux", "x86_64"): "manylinux-x86_64",
    ("linux", "aarch64"): "manylinux-aarch64",
    ("linux", "x86_64-unknown-linux-musl"): "musllinux-x86_64",
    ("linux", "aarch64-unknown-linux-musl"): "musllinux-aarch64",
    ("macos", "aarch64"): "macos-aarch64",
    ("windows", "x86_64"): "windows-x86_64",
    ("windows", "aarch64"): "windows-aarch64",
}

PIN_KEYS = {"url", "sha256", "bytes"}


class PinError(Exception):
    pass


def check_pins(pins: object) -> dict[str, object]:
    """Returns the pins if their shape is right, raises PinError otherwise."""
    if not isinstance(pins, dict):
        raise PinError("the top level is not an object")
    if set(pins) != {"ffmpeg", "majors", "platforms"}:
        raise PinError("the keys are not exactly ffmpeg, majors and platforms")
    if not isinstance(pins["ffmpeg"], str) or not pins["ffmpeg"]:
        raise PinError("ffmpeg is not a non-empty string")
    majors = pins["majors"]
    if not isinstance(majors, dict) or set(majors) != set(LIBRARIES):
        raise PinError(f"majors does not name exactly {', '.join(LIBRARIES)}")
    for lib, major in majors.items():
        if not isinstance(major, int) or isinstance(major, bool) or major <= 0:
            raise PinError(f"majors.{lib} is not a positive integer")
    platforms = pins["platforms"]
    if not isinstance(platforms, dict):
        raise PinError("platforms is not an object")
    known = set(PLATFORMS.values())
    for platform, pin in platforms.items():
        where = f"platforms.{platform}"
        if platform not in known:
            raise PinError(f"{where}: not one of {', '.join(sorted(known))}")
        if not isinstance(pin, dict) or set(pin) != PIN_KEYS:
            raise PinError(f"{where}: the keys are not exactly url, sha256 and bytes")
        if not isinstance(pin["url"], str) or not pin["url"].startswith("https://"):
            raise PinError(f"{where}.url is not an https URL")
        if not isinstance(pin["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]):
            raise PinError(f"{where}.sha256 is not 64 lowercase hex digits")
        size = pin["bytes"]
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise PinError(f"{where}.bytes is not a positive integer")
    return pins


def load_pins() -> dict[str, object]:
    try:
        return check_pins(json.loads(PINS.read_text(encoding="utf-8")))
    except json.JSONDecodeError as err:
        raise PinError(f"not JSON: {err}") from err


def platform_for(target: str, runner_os: str) -> str:
    key = (runner_os.lower(), target)
    if key not in PLATFORMS:
        known = ", ".join(f"{t} on {o}" for o, t in PLATFORMS)
        raise SystemExit(f"fetch_av.py: no platform for {target} on {runner_os}; known: {known}")
    return PLATFORMS[key]


def library_name(platform: str, lib: str, major: int) -> str:
    if platform.startswith("windows-"):
        return f"{lib}-{major}.dll"
    if platform.startswith("macos-"):
        return f"lib{lib}.{major}.dylib"
    return f"lib{lib}.so.{major}"


def download(url: str, sha256: str, size: int, path: pathlib.Path) -> None:
    """Downloads url to path, checking its byte count and sha256."""
    for attempt in range(1, 4):
        try:
            digest = hashlib.sha256()
            count = 0
            with urllib.request.urlopen(url, timeout=60) as response, path.open("wb") as out:
                while chunk := response.read(1 << 20):
                    count += len(chunk)
                    if count > size:
                        raise SystemExit(f"fetch_av.py: {url} exceeds the pinned {size} bytes")
                    digest.update(chunk)
                    out.write(chunk)
            break
        except OSError as err:
            if attempt == 3:
                raise SystemExit(f"fetch_av.py: {url}: {err}") from err
            print(f"download attempt {attempt} failed ({err}); retrying")
            time.sleep(5 * attempt)
    if count != size:
        raise SystemExit(f"fetch_av.py: {url} is {count} bytes, pinned {size}")
    if digest.hexdigest() != sha256:
        raise SystemExit(f"fetch_av.py: {url} has sha256 {digest.hexdigest()}, pinned {sha256}")


def extract(
    archive: pathlib.Path, names: list[str], optional: list[str], dest: pathlib.Path
) -> list[str]:
    """Writes each named regular file from the archive into dest, by base name.

    Every name in `names` must be there; one in `optional` is taken where it
    is. Returns what was written, in order.
    """
    found: dict[str, tarfile.TarInfo] = {}
    wanted = names + optional
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            base = member.name.rsplit("/", 1)[-1]
            if base in wanted and member.isfile():
                if base in found:
                    raise SystemExit(f"fetch_av.py: {archive.name} holds {base} twice")
                found[base] = member
        missing = [n for n in names if n not in found]
        if missing:
            raise SystemExit(f"fetch_av.py: {archive.name} has no {', '.join(missing)}")
        written = [n for n in wanted if n in found]
        for name in written:
            source = tar.extractfile(found[name])
            assert source is not None
            with source, (dest / name).open("wb") as out:
                shutil.copyfileobj(source, out)
    return written


def replace_dest() -> None:
    shutil.rmtree(DEST, ignore_errors=True)
    DEST.mkdir()


def main() -> int:
    parser = argparse.ArgumentParser(description="Place the pinned libav libraries for a wheel.")
    parser.add_argument("--target", help="the maturin target, as release.yml's matrix spells it")
    parser.add_argument("--os", default=os.environ.get("RUNNER_OS"), help="Linux, macOS or Windows")
    parser.add_argument("--check", action="store_true", help="only verify av-pins.json's shape")
    args = parser.parse_args()

    try:
        pins = load_pins()
    except (OSError, PinError) as err:
        print(f"fetch_av.py: {PINS.name}: {err}", file=sys.stderr)
        return 1
    platforms = pins["platforms"]
    assert isinstance(platforms, dict)
    if args.check:
        print(f"{PINS.name}: ok, {len(platforms)} platform(s) pinned")
        return 0
    if not args.target or not args.os:
        parser.error("--target and --os (or RUNNER_OS) are required")

    platform = platform_for(args.target, args.os)
    pin = platforms.get(platform)
    replace_dest()
    if pin is None:
        line = f"no libav pinned for {platform}; the wheel ships without codecs"
        (DEST / "README").write_text(line + "\n", encoding="utf-8")
        print(line)
        return 0

    majors = pins["majors"]
    assert isinstance(majors, dict)
    libraries = [library_name(platform, lib, majors[lib]) for lib in LIBRARIES]
    with tempfile.TemporaryDirectory() as tmp:
        archive = pathlib.Path(tmp) / pin["url"].rsplit("/", 1)[-1]
        download(pin["url"], pin["sha256"], pin["bytes"], archive)
        placed = extract(archive, libraries + list(EXTRAS), list(NOTICES), DEST)

    ffmpeg = f"ffmpeg {pins['ffmpeg']}"
    version = (DEST / "VERSION").read_text(encoding="utf-8").splitlines()
    if not version or version[0] != ffmpeg:
        raise SystemExit(f"fetch_av.py: the archive's VERSION does not name {ffmpeg}")
    for name in libraries:
        (DEST / name).chmod(0o755)

    print(f"placed {platform} libav from {pin['url']} in {DEST}:")
    for name in placed:
        print(f"  {name} ({(DEST / name).stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
