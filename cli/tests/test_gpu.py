"""The GPU adapter a run names: what reaches each sidecar's argv, and how a
spec the sidecar cannot match is refused."""

from __future__ import annotations

import os

import pytest

from ffrwd import cli, gpu, wasm
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.processes import EffectGrant, SidecarProcess
from ffrwd.wasm import Described

MODULE = "modules/gpu_box.wasm"

GRANTED = SidecarProcess(
    id="sidecar0",
    module=MODULE,
    node="n",
    grants=(EffectGrant(effect="gpu", module=MODULE),),
)
# A region decoding or encoding itself works on the device where it holds its
# pictures there.
DECODING = SidecarProcess(id="sidecar1", module=MODULE, node="n", av=True)
PLAIN = SidecarProcess(id="sidecar2", module=MODULE, node="n")


def _without(argv: list[str]) -> list[str]:
    """`argv` with its ``-gpu-adapter`` pair taken out."""
    if gpu.ADAPTER_FLAG not in argv:
        return argv
    at = argv.index(gpu.ADAPTER_FLAG)
    return argv[:at] + argv[at + 2 :]


def _spelled(argv: list[str]) -> str | None:
    if gpu.ADAPTER_FLAG not in argv:
        return None
    return argv[argv.index(gpu.ADAPTER_FLAG) + 1]


def test_a_named_adapter_reaches_each_sidecar_that_opens_the_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flag, else the variable, is spelled for a sidecar holding a GPU
    module or an av node and for a compile-time call granted the GPU, right
    after the grants, and nowhere else; with neither, no argv moves."""
    monkeypatch.setattr(wasm.binaries, "ffrwd_wasm_path", lambda: "ffrwd-wasm")
    monkeypatch.setattr(wasm.av, "spawn_args", lambda: [])
    checked: list[str | None] = []
    monkeypatch.setattr(gpu, "check", lambda: checked.append(gpu.spec()))
    described = Described(world="ffrwd:av@0.17.0", name="p", gpu=True)
    renders = (wasm.shown_argv, wasm.sidecar_argv)

    monkeypatch.delenv(gpu.ADAPTER_VAR, raising=False)
    bare = {(render, p.id): render(p) for render in renders for p in (GRANTED, DECODING, PLAIN)}
    assert not any(gpu.ADAPTER_FLAG in argv for argv in bare.values())
    assert wasm._grant_args(described, MODULE) == ["-gpu", MODULE]

    for env, flag, want in (
        ("1", None, "1"),
        (None, "radeon", "radeon"),
        ("1", "4090", "4090"),
        ("  ", None, None),
    ):
        if env is None:
            monkeypatch.delenv(gpu.ADAPTER_VAR, raising=False)
        else:
            monkeypatch.setenv(gpu.ADAPTER_VAR, env)
        checked.clear()
        with gpu.chosen(flag):
            for render in renders:
                for process in (GRANTED, DECODING, PLAIN):
                    argv = render(process)
                    expected = None if process is PLAIN else want
                    assert _spelled(argv) == expected, (render.__name__, process.id, env, flag)
                    assert _without(argv) == bare[(render, process.id)]
                    if process is GRANTED and want is not None:
                        at = argv.index(gpu.ADAPTER_FLAG)
                        assert argv[at - 2 : at] == ["-gpu", MODULE]
            assert _spelled(wasm._grant_args(described, MODULE)) == want
        # Only a run's argv is checked against what the sidecar lists, once
        # per process opening the device.
        assert checked == [want, want]
        assert os.environ.get(gpu.ADAPTER_VAR) == env


SIDECAR_SAYS = (
    "-gpu-adapter radeon: no adapter's name contains 'radeon'; the adapters are "
    "0 NVIDIA GeForce RTX 4090 (dx12), 1 Microsoft Basic Render Driver (dx12); "
    "`ffrwd-wasm --gpu-info` lists them"
)


def test_a_spec_the_sidecar_refuses_is_refused_as_the_run_spelled_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapters = (
        gpu.Adapter(0, "NVIDIA GeForce RTX 4090", "dx12", 0x10DE, 0x2684, "discrete", True),
        gpu.Adapter(1, "Microsoft Basic Render Driver", "dx12", 0x1414, 0x8C, "software", False),
    )
    asked: list[str | None] = []

    def info(given: str | None = None) -> gpu.Info:
        asked.append(given)
        if given == "radeon":
            return gpu.Info(adapters=adapters, refused=SIDECAR_SAYS)
        return gpu.Info(adapters=adapters, chosen=1)

    monkeypatch.setattr(gpu, "info", info)
    listed = (
        "no adapter's name contains 'radeon'; the adapters are 0 NVIDIA GeForce "
        "RTX 4090 (dx12), 1 Microsoft Basic Render Driver (dx12)"
    )
    monkeypatch.setenv(gpu.ADAPTER_VAR, "radeon")
    for flag, message in (
        (None, f"FFRWD_GPU_ADAPTER=radeon: {listed}"),
        ("radeon", f"--gpu-adapter radeon: {listed}"),
    ):
        with gpu.chosen(flag), pytest.raises(FfrwdError) as caught:
            gpu.check()
        assert caught.value.code is ErrorCode.RUNTIME_NOT_FOUND
        assert caught.value.message == message
        assert caught.value.hint == "`ffrwd setup gpu` lists the adapters this machine has"
    with gpu.chosen("basic"):
        gpu.check()
    monkeypatch.delenv(gpu.ADAPTER_VAR)
    gpu.check()
    assert asked == ["radeon", "radeon", "basic"]


def test_the_sidecars_list_is_read_and_a_wrong_one_refused() -> None:
    read = gpu._info_document(
        {
            "adapters": [
                {
                    "index": 0,
                    "name": "llvmpipe",
                    "backend": "vulkan",
                    "vendor": 65541,
                    "device": 0,
                    "type": "software",
                    "default": True,
                }
            ],
            "spec": "0",
            "chosen": 0,
            "refused": None,
        }
    )
    assert read == gpu.Info(
        adapters=(gpu.Adapter(0, "llvmpipe", "vulkan", 65541, 0, "software", True),),
        chosen=0,
    )
    for wrong in ({}, {"adapters": [{"index": 0}]}, {"adapters": ["x"]}, []):
        with pytest.raises(FfrwdError, match="not an adapter list"):
            gpu._info_document(wrong)


def test_a_remote_run_names_no_adapter(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", "SELECT 1", "--remote", "--gpu-adapter", "0"]) == 2
    err = capsys.readouterr().err
    assert "error: run: --gpu-adapter names an adapter on this machine" in err
    assert "hint: drop --gpu-adapter, or drop --remote" in err


def test_setup_gpu_prints_the_adapters_and_the_one_a_run_opens(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    adapters = (
        gpu.Adapter(0, "NVIDIA GeForce RTX 4090", "dx12", 0x10DE, 0x2684, "discrete", True),
        gpu.Adapter(1, "Microsoft Basic Render Driver", "dx12", 0x1414, 0x8C, "software", False),
    )
    named = {None: None, "basic": 1, "1": 1}
    monkeypatch.setattr(
        gpu,
        "info",
        lambda given=None: gpu.Info(
            adapters=adapters,
            chosen=named.get(given),
            refused=None if given in named else SIDECAR_SAYS,
        ),
    )
    table = (
        " index | name                          | backend | type     | vendor | device | default\n"
        "-------+-------------------------------+---------+----------+--------+--------+---------\n"
        " 0     | NVIDIA GeForce RTX 4090       | dx12    | discrete | 0x10de | 0x2684 | yes\n"
        " 1     | Microsoft Basic Render Driver | dx12    | software | 0x1414 | 0x008c |\n"
        "(2 rows)\n"
    )
    assert cli.main(["setup", "gpu"]) == 0
    assert capsys.readouterr().out == table + (
        "a run opens 0, NVIDIA GeForce RTX 4090 (dx12); "
        "--gpu-adapter or FFRWD_GPU_ADAPTER names another\n"
    )
    monkeypatch.setenv(gpu.ADAPTER_VAR, "basic")
    assert cli.main(["setup", "gpu"]) == 0
    assert capsys.readouterr().out == table + (
        "FFRWD_GPU_ADAPTER=basic names 1, Microsoft Basic Render Driver (dx12): a run opens it\n"
    )
    monkeypatch.setenv(gpu.ADAPTER_VAR, "radeon")
    assert cli.main(["setup", "gpu"]) == 1
    said = capsys.readouterr()
    assert said.out == table
    assert "RUNTIME_NOT_FOUND: FFRWD_GPU_ADAPTER=radeon: no adapter's name" in said.err
    assert cli.main(["setup", "gpu", "--cuda"]) == 2
