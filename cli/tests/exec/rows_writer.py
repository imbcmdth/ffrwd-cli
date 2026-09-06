"""A stand-in for the sidecar, for the exec tier: one WebVTT document per pipe.

Run as ``python rows_writer.py [--compute=SECONDS] <cues> <path> [<path> ...]``.
Like the sidecar, it opens every pipe up front and closes none of them until it
has written them all -- so the process reading the first document sees no end of
it while the second is being written.

``--compute`` spends that many seconds of CPU before each document after the
first, which is what a sidecar working over what it has read looks like: nothing
crosses any pipe while it does.
"""

from __future__ import annotations

import sys
import time


def document(cues: int, mark: str) -> bytes:
    lines = ["WEBVTT", ""]
    for i in range(cues):
        stamp = f"{i // 60:02d}:{i % 60:02d}"
        lines += [f"{stamp}.000 --> {stamp}.500", f"{mark} {'x' * 200}", ""]
    return "\n".join(lines).encode("utf-8")


def compute(seconds: float) -> None:
    """Use CPU, and nothing else, for `seconds`."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        pass


def main(argv: list[str]) -> None:
    seconds = 0.0
    if argv and argv[0].startswith("--compute="):
        seconds = float(argv[0].partition("=")[2])
        argv = argv[1:]
    cues, paths = int(argv[0]), argv[1:]
    pipes = [open(path, "wb") for path in paths]
    try:
        for index, pipe in enumerate(pipes):
            if index and seconds:
                compute(seconds)
            pipe.write(document(cues, f"cue{index}"))
            pipe.flush()
    finally:
        for pipe in pipes:
            pipe.close()


if __name__ == "__main__":
    main(sys.argv[1:])
