# SPDX-License-Identifier: MIT
"""`compass trace ops` on DeepSeek-V4-Pro at one rank of the nightly DPA-TP8
deployment, EP off and EP8, prefill and decode, with no device.

A step under DP is traced at its concrete width and keyed by it; every peer
runs this rank's width, so what the MoE sees is eight ranks' rows.
"""

import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest
from test_capture_ir import _decoder, _ops

from atom.compass.artifacts import ArtifactStore, Kind, Topology

TREE = pathlib.Path(__file__).resolve().parents[2]
# `deepseek-ai/DeepSeek-V4-Pro` at revision
# b5968e9190ef611bbf34a7229255be88a0e937c1, its config.json byte for byte.
CONFIG = pathlib.Path(__file__).with_name("deepseek_v4_pro_config.json")
CONFIG_SHA256 = "5fe4568daee51c208cb8a79538eaeda090ae011ade1dee2c386aa95f569c810e"
LAYERS = json.loads(CONFIG.read_text())["num_hidden_layers"]
DPA_TP8 = ["--kv_cache_dtype", "fp8", "--enable-dp-attention"]
EP8 = ["--enable-expert-parallel"]
DP = 8

_CAPTURES: dict = {}


def capture(tmp_path_factory, *argv):
    """The structure, graph and declarations `trace ops` published for `argv`."""
    if argv in _CAPTURES:
        return _CAPTURES[argv]
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == CONFIG_SHA256
    root = tmp_path_factory.mktemp("v4pro")
    (root / "model").mkdir()
    (root / "model" / "config.json").write_bytes(CONFIG.read_bytes())
    command = ["trace", "ops", "--model", str(root / "model"), "--tp", "8"]
    command += ["--arch", "gfx950", "--store", str(root / "store"), *argv]
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from atom.compass.cli import main; main(sys.argv[1:])",
            *command,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
        cwd=TREE,
        env={**os.environ, "PYTHONPATH": str(TREE)},
    )
    published = re.search(r"^published .*'(.*)'\) as (\w+)$", done.stdout, re.MULTILINE)
    assert published, f"rc {done.returncode}\n{done.stderr[-4000:]}"
    store = ArtifactStore(str(root / "store"))
    entry = store.read(store.key_from_token(Kind.OP_GRAPH, published.group(2)))
    rank = next(iter(Topology().ranks()))
    graph, declared = (
        json.loads(entry.read_member(stem, rank, "json"))
        for stem in ("graph", "declared")
    )
    _CAPTURES[argv] = published.group(1), graph, declared
    return _CAPTURES[argv]


def _pars(region):
    """Every `Par` the step runs, repeats expanded, each as its branches' op names."""
    if "par" in region:
        return [[[op["op"] for op in _ops(b)] for b in region["par"]]]
    if "repeat" in region:
        return _pars(region["repeat"]) * region["count"]
    return [par for item in region.get("seq", []) for par in _pars(item)]


STEPS = {
    "decode": (["--width", "2"], "decode B=2", 2),
    "prefill": (["--width", "512", "--prefill"], "prefill T=512 B=1", 512),
}


@pytest.mark.parametrize("ep", [False, True], ids=["ep_off", "ep8"])
@pytest.mark.parametrize("step", STEPS)
def test_each_layer_traces_concrete_with_the_shared_expert_beside_the_moe(
    step, ep, tmp_path_factory
):
    argv, shape, rows = STEPS[step]
    structure, graph, declared = capture(
        tmp_path_factory, *argv, "--", *DPA_TP8, *(EP8 if ep else [])
    )

    assert f" tp=1 dp={DP} ep={int(ep)} {shape} eager gfx950" in structure
    assert graph["guards"] == [] and declared["host_resolutions"] == []
    every = [d for op in _ops(graph["region"]) for s in op["in"] + op["out"] for d in s]
    assert all(isinstance(d, int) for d in every)
    # Every layer's shared expert runs on a side stream beside its routed MoE.
    pars = _pars(graph["region"])
    assert len(pars) == LAYERS
    for routed, shared in pars:
        assert "aiter.fused_moe_.default" in routed
        assert "aiter.silu_and_mul.default" in shared
    layers = _decoder(graph)
    assert len(layers) == LAYERS
    for layer in layers:
        names = [op["op"] for op in layer]
        assert names.count("aiter.v4_attention_with_output.default") == 1
        assert names.count("aiter.fused_moe_.default") == 1
        exchanged = [op for op in layer if op["kind"] == "declared"]
        if ep:
            assert [op["op"] for op in exchanged] == ["mori.dispatch", "mori.combine"]
        else:
            assert {op["op"].split(".")[0] for op in exchanged} == {"dp"}
        # In: this rank's rows. Out of the dispatch or gather: every rank's.
        assert exchanged[0]["in"][0][0] == rows
        assert exchanged[0]["out"][0][0] == DP * rows


def test_a_dp_decode_traces_at_width_1_which_a_symbol_cannot(tmp_path_factory):
    structure, graph, _ = capture(tmp_path_factory, "--width", "1", "--", *DPA_TP8)

    assert " decode B=1 eager " in structure
    rows = {
        (op["op"], op["out"][0][0])
        for op in _ops(graph["region"])
        if op["op"].startswith("dp.")
    }
    assert rows == {
        ("dp.all_gather_with_padding", DP),
        ("dp.reduce_scatter_with_unpadding", 1),
    }
