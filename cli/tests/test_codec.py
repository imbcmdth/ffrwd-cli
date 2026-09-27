"""Codec packages: an encoder written where a codec name goes, a decoder an
input() inserts over a stream carrying the tag it reads."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlglot import exp

from ffrwd import wasm
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.parser import parse, resolve
from ffrwd.project import discover


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
