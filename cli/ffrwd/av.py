"""The codecs a sidecar region decodes and encodes itself, and where its
libraries are.

A region whose entry function says ``SET ffrwd.edges = 'coded'`` may read
its input as the coded packets an input carries and hand its output back
coded: the sidecar's own ``decode`` and ``encode`` nodes run libavcodec
inside the region, and the ffmpegs on either side copy the stream.
:data:`DECODABLE` and :data:`HOSTED_ENCODERS` are what the bundled LGPL
build does, :data:`ENCODE_OPTIONS` how a destination's video options are
spelled on the ``encode`` node, and :func:`spawn_args` the ``-av-runtime``
a spawned sidecar is told. :func:`sidecar_filters` is what the sidecar's
libavfilter has, which decides the filters a region runs on its device.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
from collections.abc import Mapping

from . import binaries
from .ir import DECODE, ENCODE, Graph

__all__ = [
    "AV_RUNTIME_FLAG",
    "DECODABLE",
    "ENCODE_OPTIONS",
    "HARDWARE_ENCODER_SUFFIXES",
    "HOSTED_ENCODERS",
    "RUNTIME_DIR_VAR",
    "sessions",
    "sidecar_filters",
    "spawn_args",
]

# The codecs the bundled libavcodec decodes.
DECODABLE = frozenset(
    {"h264", "hevc", "av1", "vp9", "mpeg2video", "prores", "mjpeg", "ffv1", "mpeg4"}
)

# The encoders the bundled libavcodec has, by ffmpeg's own names.
HOSTED_ENCODERS = frozenset(
    {
        "h264_nvenc",
        "hevc_nvenc",
        "av1_nvenc",
        "h264_amf",
        "hevc_amf",
        "av1_amf",
        "h264_d3d12va",
        "hevc_d3d12va",
        "h264_mf",
        "hevc_mf",
        "h264_videotoolbox",
        "hevc_videotoolbox",
        "h264_vaapi",
        "hevc_vaapi",
        "av1_vaapi",
        "mpeg4",
        "mpeg2video",
        "mjpeg",
        "ffv1",
    }
)

# A destination's video option -> the name the ``encode`` node takes it by,
# which is ffmpeg's own.
ENCODE_OPTIONS: Mapping[str, str] = {
    "crf": "crf",
    "preset": "preset",
    "gop": "g",
    "video_bitrate": "b",
    "maxrate": "maxrate",
    "bufsize": "bufsize",
    "keyint_min": "keyint_min",
    "profile": "profile",
    "level": "level",
    "tune": "tune",
    "sc_threshold": "sc_threshold",
    "pix_fmt": "pix_fmt",
}

# The encoder names that run on a hardware encode session.
HARDWARE_ENCODER_SUFFIXES = ("_nvenc", "_qsv", "_vaapi", "_amf", "_d3d12va")

# The ``encode`` node's argument naming its encoder.
CODEC = "codec"

# The flag naming the libraries' directory, and the variable a developer
# names their own with.
AV_RUNTIME_FLAG = "-av-runtime"
RUNTIME_DIR_VAR = "FFRWD_AV_RUNTIME"


def spawn_args() -> list[str]:
    """The ``-av-runtime`` a spawned sidecar is given.

    Empty when the environment already names a directory, which keeps a
    developer's own `FFRWD_AV_RUNTIME` in charge; empty too when the
    installed wheel carries none, so the sidecar looks beside itself and
    gives its own refusal.
    """
    named = os.environ.get(RUNTIME_DIR_VAR)
    if named and named.strip():
        return []
    directory = binaries.ffrwd_av_dir()
    if directory is None:
        return []
    return [AV_RUNTIME_FLAG, directory]


# How long the compile waits for the sidecar to say what its libav has.
_AV_INFO_SECONDS = 30.0


@functools.cache
def sidecar_filters() -> frozenset[str] | None:
    """The filters the sidecar's libavfilter has, by name, as
    ``ffrwd-wasm --av-info`` lists them; None where that cannot be known.

    Asked once per process, with the ``-av-runtime`` a run would be given.
    Empty where the sidecar loads its libav and that has no libavfilter (a
    wheel from before it was bundled). None where no sidecar is found, it
    does not answer, or it loads no libav at all, which is a compile on a
    machine that is not the one the command runs on: the caller then writes
    the command for the reader's sidecar, as the card pass does for an
    ffmpeg it cannot see.
    """
    sidecar = binaries.ffrwd_wasm_path()
    if sidecar is None:
        return None
    try:
        done = subprocess.run(
            [sidecar, "--av-info", *spawn_args()],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=_AV_INFO_SECONDS,
            check=False,
        )
        info = json.loads(done.stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    if not isinstance(info, dict) or info.get("error") is not None:
        return None
    filters = info.get("filters")
    if not isinstance(filters, list):
        return frozenset()
    return frozenset(name for name in filters if isinstance(name, str))


def sessions(graph: Graph | None) -> tuple[int, int]:
    """The hardware encode and decode sessions a region's graph holds.

    One encode per ``encode`` node naming a hardware encoder, one decode per
    ``decode`` node.
    """
    if graph is None:
        return 0, 0
    encodes = sum(
        1
        for node in graph.nodes.values()
        if node.filter == ENCODE
        and str(node.args.get(CODEC, "")).endswith(HARDWARE_ENCODER_SUFFIXES)
    )
    decodes = sum(1 for node in graph.nodes.values() if node.filter == DECODE)
    return encodes, decodes
