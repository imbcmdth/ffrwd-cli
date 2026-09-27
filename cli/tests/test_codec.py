"""Codec packages: an encoder written where a codec name goes, a decoder an
input() inserts over a stream carrying the tag it reads."""

from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest
from sqlglot import exp

from ffrwd import wasm
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.ir import Graph
from ffrwd.lower import lower
from ffrwd.parser import parse, resolve
from ffrwd.probe import ProbeResult, StreamMeta
from ffrwd.project import discover
from ffrwd.registry import Registry, load_reference
from ffrwd.split import insert_splits
from ffrwd.wasm import Described

SNAPSHOT_PATH = Path(__file__).resolve().parent / "data" / "reference_registry.json"


def _codec_package(root: Path) -> None:
    """A package declaring an encoder and a decoder over one module."""
    (root / "src").mkdir(parents=True)
    (root / "src" / "codec.sql").write_text(
        "CREATE FUNCTION encode(bitrate number DEFAULT 200000000)\n"
        "  RETURNS encoder AS 'modules/codec.wasm', 'encode' LANGUAGE wasm;\n"
        "CREATE FUNCTION decode() RETURNS decoder\n"
        "  AS 'modules/codec.wasm', 'decode' LANGUAGE wasm;\n",
        encoding="utf-8",
    )
    (root / "ffrwd.json").write_text(
        json.dumps(
            {
                "name": "ffrwd/codec",
                "version": "1.0.0",
                "lib": {"encode": "src/codec.sql", "decode": "src/codec.sql"},
            }
        ),
        encoding="utf-8",
    )


# -- declarations -------------------------------------------------------------


def test_an_encoder_and_a_decoder_are_declared_by_their_return() -> None:
    resolved = resolve(
        parse(
            "CREATE FUNCTION enc(bitrate number DEFAULT 1) RETURNS encoder "
            "AS 'm.wasm', 'encode' LANGUAGE wasm;\n"
            "CREATE FUNCTION dec() RETURNS decoder AS 'm.wasm', 'decode' LANGUAGE wasm;\n"
            "COPY (SELECT f.video[1] FROM input('a.nut', decoder => dec()) f) "
            "TO 'b.nut' WITH (video_codec enc(bitrate => 5))"
        )
    )
    enc, dec = resolved.wasm["enc"], resolved.wasm["dec"]
    assert (enc.is_encoder, enc.is_decoder, enc.is_value) == (True, False, False)
    assert (dec.is_encoder, dec.is_decoder, dec.is_value) == (False, True, False)


def test_a_codec_takes_no_stream_parameter() -> None:
    with pytest.raises(FfrwdError) as caught:
        parse_and_resolve = resolve(
            parse(
                "CREATE FUNCTION enc(v video_stream) RETURNS encoder "
                "AS 'm.wasm', 'encode' LANGUAGE wasm;\n"
                "COPY (SELECT f.video[1] FROM input('a.nut') f) TO 'b.nut' "
                "WITH (video_codec enc(f.video[1]))"
            )
        )
        del parse_and_resolve
    assert caught.value.code is ErrorCode.UNSUPPORTED_SQL
    assert "declares the stream parameter 'v'" in caught.value.message
    assert "takes only values" in (caught.value.hint or "")


# -- the spelling ---------------------------------------------------------------


@pytest.mark.parametrize(
    "option",
    [
        "video_codec ffrwd.codec.encode(bitrate => 5)",
        "video_codec => ffrwd.codec.encode(bitrate => 5)",
    ],
)
def test_a_packages_encoder_is_written_where_a_codec_name_goes(
    tmp_path: Path, option: str
) -> None:
    _codec_package(tmp_path)
    packages = discover(tmp_path)
    assert packages is not None
    resolved = resolve(
        parse(f"COPY (SELECT f.video[1] FROM input('a.nut') f) TO 'b.nut' WITH ({option})"),
        packages=packages,
    )
    assert list(resolved.wasm) == ["ffrwd.codec.encode"]
    (sink,) = resolved.sinks
    (codec,) = [o for o in sink.options if o.name == "video_codec"]
    assert isinstance(codec.value, exp.Anonymous)
    assert codec.value.name == "ffrwd.codec.encode"


def test_a_packages_decoder_is_named_in_an_inputs_options(tmp_path: Path) -> None:
    _codec_package(tmp_path)
    packages = discover(tmp_path)
    assert packages is not None
    resolved = resolve(
        parse(
            "COPY (SELECT f.video[1] FROM input('a.nut', decoder => ffrwd.codec.decode()) f) "
            "TO 'b.y4m'"
        ),
        packages=packages,
    )
    assert list(resolved.wasm) == ["ffrwd.codec.decode"]


# -- what a codec module describes ------------------------------------------------


def test_a_codec_modules_describe_is_read() -> None:
    described = wasm._described(
        "m.wasm",
        {
            "world": "ffrwd:av@0.18.0",
            "name": "pyrowave",
            "pixel_formats": ["yuv420p"],
            "encoder": {"codec": "pyrowave", "fourcc": "PYRW", "delay": 0},
            "decoder": {"fourccs": ["PYRW"]},
        },
    )
    assert described.encoder == wasm.EncoderInfo(codec="pyrowave", fourcc="PYRW")
    assert described.decoder == wasm.DecoderInfo(fourccs=("PYRW",))
    assert wasm.hosts_codec(described.world)
    assert not wasm.hosts_codec("ffrwd:av@0.17.0")
    plain = wasm._described("m.wasm", {"world": "ffrwd:av@0.18.0", "name": "p"})
    assert plain.encoder is None and plain.decoder is None


# -- lowering an encoder ------------------------------------------------------------

CODEC = "modules/codec.wasm"
_ENCODER = (
    "CREATE FUNCTION enc(bitrate number DEFAULT 1000) RETURNS encoder "
    f"AS '{CODEC}', 'encode' LANGUAGE wasm;\n"
)


def _codec(kind: str = "video") -> Described:
    formats = {"pixel_formats": ("yuv420p",)} if kind == "video" else {"sample_formats": ("f32",)}
    return Described(
        world="ffrwd:av@0.18.0",
        name="codec",
        params_schema={"type": "object", "properties": {"bitrate": {"type": "number"}}},
        encoder=wasm.EncoderInfo(codec="testcodec", fourcc="FTST"),
        decoder=wasm.DecoderInfo(fourccs=("FTST",)),
        **formats,  # type: ignore[arg-type]
    )


@functools.cache
def _registry() -> Registry:
    return load_reference(SNAPSHOT_PATH)


def _clip() -> dict[str, ProbeResult | None]:
    return {
        "f": ProbeResult(
            streams=[
                StreamMeta(
                    type="video", index=0, metadata={}, width=64, height=48,
                    fps="10/1", sample_rate=None, codec="h264",
                ),
                StreamMeta(
                    type="audio", index=0, metadata={}, width=None, height=None,
                    fps=None, sample_rate=48000, codec="aac", channels=2,
                ),
            ]
        )
    }


def _lowered(query: str, codec: Described | None = None) -> Graph:
    return insert_splits(
        lower(
            resolve(parse(_ENCODER + query)),
            _clip(),
            registry=_registry(),
            describes={CODEC: codec or _codec()},
        )
    )


def _refused(query: str, codec: Described | None = None) -> FfrwdError:
    with pytest.raises(FfrwdError) as caught:
        _lowered(query, codec)
    return caught.value


def test_an_encoder_codes_the_outputs_stream_and_the_file_copies_it() -> None:
    graph = _lowered(
        "COPY (SELECT f.video[1], f.audio[1] FROM input('clip.mp4') f) TO 'out.nut' "
        "WITH (video_codec enc(bitrate => 5000), audio_codec 'aac')"
    )
    (encoder,) = graph.encoders
    node = graph.nodes[encoder]
    assert (node.filter, node.args, node.inputs) == (CODEC, {"bitrate": 5000}, ["src:f:v:0"])
    (unit,) = graph.sinks
    assert [o.ref for o in unit.outputs] == [encoder, "src:f:a:0"]
    assert "video_codec" not in unit.options
    assert unit.options["audio_codec"] == "aac"


@pytest.mark.parametrize("path", ["out.mkv", "out.mov", "out.nut"])
def test_a_container_that_keeps_a_tag_takes_the_stream(path: str) -> None:
    graph = _lowered(
        f"COPY (SELECT f.video[1] FROM input('clip.mp4') f) TO '{path}' "
        "WITH (video_codec enc())"
    )
    assert len(graph.encoders) == 1


@pytest.mark.parametrize(
    ("query", "needle"),
    [
        (
            "COPY (SELECT f.video[1] FROM input('clip.mp4') f) TO 'out.mp4' "
            "WITH (video_codec enc())",
            "'out.mp4' is mp4, and it cannot hold testcodec",
        ),
        (
            "COPY (SELECT f.video[1] FROM input('clip.mp4') f) TO 'out.nut' "
            "WITH (video_codec enc(), crf 20, gop 30)",
            "'enc' encodes the video itself, so crf, gop have nothing to shape",
        ),
        (
            "COPY (SELECT f.video[1] FROM input('clip.mp4') f) TO 'out.nut' "
            "WITH (audio_codec enc())",
            "'enc' encodes video, and audio_codec names the audio codec",
        ),
        (
            "COPY (SELECT f.audio[1] FROM input('clip.mp4') f) TO 'out.nut' "
            "WITH (video_codec enc())",
            "'out.nut' is written no video stream",
        ),
        (
            "COPY (SELECT f.video[1] FROM input('clip.mp4') f) TO 'out.nut' "
            "WITH (crf enc())",
            "'enc' is an encoder, which names a stream's codec, and 'crf' is not a codec option",
        ),
    ],
)
def test_an_encoder_with_no_reading_is_refused(query: str, needle: str) -> None:
    error = _refused(query)
    assert error.code is ErrorCode.UNSUPPORTED_SQL
    assert needle in error.message


def test_an_encoder_in_a_world_before_codecs_is_refused() -> None:
    old = Described(
        world="ffrwd:av@0.17.0",
        name="codec",
        pixel_formats=("yuv420p",),
        encoder=wasm.EncoderInfo(codec="testcodec", fourcc="FTST"),
    )
    error = _refused(
        "COPY (SELECT f.video[1] FROM input('clip.mp4') f) TO 'out.nut' "
        "WITH (video_codec enc())",
        old,
    )
    assert "hosted from ffrwd:av@0.18.0 on" in error.message
