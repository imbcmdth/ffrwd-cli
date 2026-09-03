"""Generate the real coded stream `source-replay` compiles in.

`sidecar/modules/source-replay` is a fleet packet source with no filesystem
access (the wasm sandbox it runs in has none), so the seven packets it hands
out through `next()` are compiled into the module rather than read from a
path. This script is what produces that compiled-in data: it encodes seven
frames of a 32x24 test pattern with libx264, B-frames on and a GOP longer
than the clip (so libx264 never emits a second keyframe), which settles into
decode order I0 P3 B1 B2 P6 B4 B5 -- a reorder depth of 2, the exact pattern
`nut::mux`'s own coded-packet roundtrip test pins. `-flags +global_header`
keeps SPS/PPS out of the per-frame bitstream and in the container's
extradata only, so the seven access units the module replays carry nothing
but their own coded bytes.

Output (checked in, small -- about 1.2KB total):

- `sidecar/modules/source-replay/src/generated/packets.bin` -- the real
  bytes: extradata (SPS+PPS, each with its own Annex-B start code) followed
  by the seven packets' coded bytes, back to back.
- `sidecar/modules/source-replay/src/generated/packets.rs` -- constants
  `lib.rs` includes with `include!`: `RAW` (the blob above, `include_bytes!`
  ed), `EXTRADATA_LEN`, `PACKET_LENS` (one length per packet, so `lib.rs`
  can slice `RAW` without re-parsing Annex-B), `PACKET_TABLE` (pts, dts,
  keyframe per packet, decode order), and the stream's actual `WIDTH`,
  `HEIGHT`, `PROFILE`, `LEVEL` as libx264 and its SPS report them.

Both pts and dts are ffprobe's own values for this encode, normalized to
start at 0 and expressed in units of one frame (dividing by the packet
duration, which is constant across an untrimmed CFR encode).

Regenerate with::

    python scripts/gen_replay_packets.py

then rebuild the module (`cargo build --release --target wasm32-wasip2 -p
source-replay` from `sidecar/modules`) and re-derive any pinned expectation
that names the old bytes/values from this script's report, never from a
failing test's output.

Stdlib only -- no third-party imports, so this script itself never needs the
`[dev]` extra installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CLI_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _CLI_ROOT.parent
_MODULE_SRC = _REPO_ROOT / "sidecar" / "modules" / "source-replay" / "src"
_GENERATED_DIR = _MODULE_SRC / "generated"

_SIZE = "32x24"
_RATE = 25
_FRAME_COUNT = 7

# bframes=2, a fixed (non-adaptive) B-frame pattern, one reference frame, and
# no scene-cut detection: what settles decode order into a flat, repeating
# I-P-B-B GOP rather than one shaped by the (blank) test pattern's content.
# `-g` well past the clip's 7 frames means libx264 never emits a second I
# frame.
_X264_PARAMS = "bframes=2:b-adapt=0:ref=1:scenecut=0"


@dataclass(frozen=True)
class ReplayPackets:
    """Everything derived from one encode, ready to write or report."""

    cmd: list[str]
    blob: bytes
    extradata: bytes
    extradata_len: int
    packet_lens: list[int]
    packet_table: list[tuple[int, int | None, bool]]
    width: int
    height: int
    level: int
    leading_none: int


def _require(tool: str) -> None:
    if shutil.which(tool) is None:
        raise SystemExit(f"error: {tool} not found on PATH")


def _run(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(args, capture_output=True, check=False)
    if result.returncode != 0:
        sys.stderr.write(result.stderr.decode("utf-8", errors="replace"))
        raise SystemExit(f"command failed: {' '.join(args)}")
    return result


def _ffprobe_json(path: Path, entries: str, *, of_streams: bool) -> dict[str, Any]:
    show = "-show_streams" if of_streams else "-show_packets"
    result = _run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v",
            show,
            "-show_entries", entries,
            "-of", "json",
            str(path),
        ]
    )
    parsed: dict[str, Any] = json.loads(result.stdout)
    return parsed


def _encode_command(out_nut: Path) -> list[str]:
    return [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc2=size={_SIZE}:rate={_RATE}",
        "-frames:v", str(_FRAME_COUNT),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-x264-params", _X264_PARAMS,
        "-g", "250",
        "-flags:v", "+global_header",
        "-f", "nut", str(out_nut),
    ]


def _generate() -> ReplayPackets:
    with tempfile.TemporaryDirectory(prefix="gen-replay-packets-") as tmp:
        tmp_dir = Path(tmp)
        out_nut = tmp_dir / "out.nut"
        cmd = _encode_command(out_nut)
        _run(cmd)

        stream = _ffprobe_json(
            out_nut, "stream=width,height,profile,level,extradata_size", of_streams=True
        )["streams"][0]
        packets: list[dict[str, Any]] = _ffprobe_json(
            out_nut, "packet=pts,dts,flags,size,duration", of_streams=False
        )["packets"]
        if len(packets) != _FRAME_COUNT:
            raise SystemExit(f"expected {_FRAME_COUNT} packets, ffprobe reported {len(packets)}")

        # `dump_extra` re-inserts the stream's extradata before the first
        # keyframe: with `-flags +global_header` that is the only place
        # SPS/PPS live, so prepending it here and slicing by size is how the
        # extradata bytes are recovered without hand-parsing NUT framing.
        dumped_path = tmp_dir / "dumped.h264"
        _run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(out_nut),
                "-c", "copy", "-bsf:v", "dump_extra=freq=keyframe",
                "-f", "h264", str(dumped_path),
            ]
        )
        dumped = dumped_path.read_bytes()

        extradata_len = int(stream["extradata_size"])
        extradata = dumped[:extradata_len]
        body = dumped[extradata_len:]

        durations = {int(p["duration"]) for p in packets}
        if len(durations) != 1:
            raise SystemExit(f"packet durations are not constant: {sorted(durations)}")
        (unit,) = durations

        def to_frames(ticks: str | None) -> int | None:
            if ticks is None:
                return None
            value = int(ticks)
            if value % unit != 0:
                raise SystemExit(f"timestamp {value} is not a multiple of the frame unit {unit}")
            return value // unit

        base = min(frames for p in packets if (frames := to_frames(p["pts"])) is not None)

        packet_lens: list[int] = []
        packet_table: list[tuple[int, int | None, bool]] = []
        cursor = 0
        for p in packets:
            size = int(p["size"])
            packet_lens.append(size)
            cursor += size

            raw_pts = to_frames(p["pts"])
            assert raw_pts is not None, "a packet's own pts is never unset"
            pts = raw_pts - base
            raw_dts = to_frames(p.get("dts"))
            dts = raw_dts - base if raw_dts is not None else None
            keyframe = "K" in p["flags"]
            packet_table.append((pts, dts, keyframe))

        if cursor != len(body):
            raise SystemExit(
                f"packet sizes sum to {cursor} bytes but the dumped body is {len(body)} bytes"
            )

        blob = extradata + body
        keyframes = sum(1 for _, _, key in packet_table if key)
        if keyframes != 1:
            raise SystemExit(f"expected exactly one keyframe, got {keyframes}")
        leading_none = 0
        for _, dts, _ in packet_table:
            if dts is not None:
                break
            leading_none += 1

        return ReplayPackets(
            cmd=cmd,
            blob=blob,
            extradata=extradata,
            extradata_len=extradata_len,
            packet_lens=packet_lens,
            packet_table=packet_table,
            width=int(stream["width"]),
            height=int(stream["height"]),
            level=int(stream["level"]),
            leading_none=leading_none,
        )


# ffprobe's `profile` entry for h264 is the human name (High/Main/...), not
# the numeric profile_idc a SPS carries, so the number is read out of the
# SPS bytes themselves -- the 5th byte of the SPS NAL (after the 4-byte
# start code and the 1-byte NAL header) is profile_idc.
def _profile_from_sps(extradata: bytes) -> int:
    # extradata = 4-byte start code + SPS NAL header + SPS payload...
    if extradata[0:4] != b"\x00\x00\x00\x01" or (extradata[4] & 0x1F) != 7:
        raise SystemExit("extradata does not start with an Annex-B SPS")
    return extradata[5]


def _packet_table_line(entry: tuple[int, int | None, bool]) -> str:
    pts, dts, keyframe = entry
    dts_repr = f"Some({dts})" if dts is not None else "None"
    return f"    ({pts}, {dts_repr}, {str(keyframe).lower()}),"


def _write_generated(result: ReplayPackets) -> None:
    _GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    (_GENERATED_DIR / "packets.bin").write_bytes(result.blob)

    profile_idc = _profile_from_sps(result.extradata)
    level_idc = result.level

    table_lines = "\n".join(_packet_table_line(entry) for entry in result.packet_table)
    lens_line = ", ".join(str(n) for n in result.packet_lens)

    rs = f"""\
// Generated by cli/scripts/gen_replay_packets.py from a real libx264 encode.
// Do not hand-edit -- regenerate with that script and see its docstring.

/// Extradata (SPS+PPS) followed by the seven packets' coded bytes, back to
/// back -- sliced apart with `EXTRADATA_LEN` and `PACKET_LENS`.
pub(crate) const RAW: &[u8] = include_bytes!("packets.bin");

pub(crate) const EXTRADATA_LEN: usize = {result.extradata_len};

/// Byte length of each of the seven packets, decode order, into `RAW` right
/// after the extradata.
pub(crate) const PACKET_LENS: &[usize] = &[{lens_line}];

/// Decode order I0 P3 B1 B2 P6 B4 B5, and the dts a reorder buffer of depth
/// {result.leading_none} settles for it -- the exact pattern and numbers
/// `nut::mux`'s own coded-packet roundtrip test pins, so this module's proof
/// rests on values already known correct.
pub(crate) const PACKET_TABLE: &[(i64, Option<i64>, bool)] = &[
{table_lines}
];

/// What the encode actually is, read from libx264's own report and the
/// SPS it wrote -- not assumed.
pub(crate) const WIDTH: u32 = {result.width};
pub(crate) const HEIGHT: u32 = {result.height};
pub(crate) const PROFILE: i32 = 0x{profile_idc:02x};
pub(crate) const LEVEL: i32 = 0x{level_idc:02x};
"""
    (_GENERATED_DIR / "packets.rs").write_text(rs, encoding="utf-8")


def main() -> int:
    _require("ffmpeg")
    _require("ffprobe")
    result = _generate()
    _write_generated(result)

    profile_idc = _profile_from_sps(result.extradata)
    level_idc = result.level
    print("encode command:")
    print("  " + " ".join(result.cmd))
    print()
    print("pts/dts/keyframe table (decode order):")
    for pts, dts, key in result.packet_table:
        print(f"  ({pts}, {dts}, {key})")
    print()
    print(f"SPS: {result.width}x{result.height}, "
          f"profile=0x{profile_idc:02x}, level=0x{level_idc:02x}")
    print(f"extradata: {result.extradata.hex()}")
    print(f"packet lengths: {result.packet_lens}")
    print()
    print(f"wrote {_GENERATED_DIR / 'packets.bin'} ({len(result.blob)} bytes)")
    print(f"wrote {_GENERATED_DIR / 'packets.rs'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
