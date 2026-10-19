# SPDX-License-Identifier: MIT
"""The step roofline over DeepSeek-V4-Pro's modules: hyper-connection mixes,
sparse attention with its indexer, the routed experts and their exchange, and
a two-batch-overlap step laid on the compute and comm streams.

Each is checked against a count by hand, on a machine with no kernel floor,
once memory-bound and once compute-bound. The published model's meta build at
DPA-TP8, expert parallelism off and at EP8, classifies every module; the build
owns its process because the driverless declarations are process-global.
"""

import collections
import copy
import json
import os
import pathlib
import subprocess
import sys
from math import comb
from types import SimpleNamespace

import pytest
import torch
from test_backend_step_roofline import ROOFLINE
from test_dp_step_max import runner
from test_moe_gather_rows import decode

from atom.compass.backends import BatchView, CostRefused, RequestShape
from atom.compass.backends.step_roofline import Module, StepRooflineBackend, overlap
from atom.compass.backends.work import (
    Work,
    activated_experts,
    allgather,
    allreduce,
    attention,
    bounds,
    gemm,
    grouped_gemm,
    max_rank_load,
)
from atom.compass.runner import projection
from atom.compass.spec import MachineSpec

TREE = pathlib.Path(__file__).resolve().parents[2]
CONFIG = pathlib.Path(__file__).with_name("deepseek_v4_pro_config.json")
#: Experts, local experts, top k, intermediate, hidden, group, DP width, split.
EP_OFF = (8, 8, 2, 16, 32, 2, 2, 0)
EP2 = (8, 4, 2, 32, 32, 2, 2, 0)
#: The link: one hop of 2 us, 1e12 B/s derated by 0.8, one lane at two ranks.
LINK = 1.0e12 * 0.80


def machine(bf16):
    document = copy.deepcopy(ROOFLINE)
    document["device"]["kernel_floor_s"] = 1e-15
    document["device"]["memory"]["bandwidth_curve"] = [[1, 1e9], [2**40, 1e9]]
    document["device"]["compute"].update(bf16_flops=bf16, fp8_flops=2 * bf16)
    return MachineSpec.from_mapping(document)


BOUND = pytest.mark.parametrize(
    "spec", [machine(1e18), machine(1e6)], ids=["memory-bound", "compute-bound"]
)


def view(*rows, moe_rows=None, tbo=None):
    return BatchView(
        tuple(RequestShape(*r) for r in rows), moe_rows=moe_rows, tbo_tokens=tbo
    )


def _module(work, spec):
    b = bounds(work, spec)
    return max(b["floor"], b["compute"], b["memory"]) + b["comm"]


def _price(modules, step, spec):
    cost = StepRooflineBackend(spec, modules).price(step)
    return {t.name: t.seconds for t in cost.terms}


def test_two_micro_batches_overlap_as_atoms_streams_order_them():
    # Three layers of compute 3, exchange in 2, experts 4, exchange out 2 for
    # each micro-batch: the timeline ATOM's own TBO ring ran, finishing at 44
    # against 66 run one after the other.
    layer = [[3, 3], [2, 2], [4, 4], [2, 2]]
    assert overlap([layer] * 3) == 44
    # One micro-batch's slow compute holds the other's experts behind it.
    assert overlap([[[5, 1], [1, 1], [1, 1], [1, 1]]]) == 9


@BOUND
def test_the_routed_experts_and_their_exchange_by_hand(spec):
    tokens, rows = 3, 10
    ep_off = Module("experts", "moe", EP_OFF, "fp4", ranks=2)
    ep2 = Module("experts", "moe", EP2, "fp4", ranks=2, act="fp8")
    one_rank = Module("experts", "moe", (8, 8, 2, 16, 32, 2, 1, 0), "fp4", ranks=2)
    step = view((tokens, tokens, False), moe_rows=rows)
    # EP off: every expert a row reaches, at a sixteenth of the intermediate;
    # the gathered hidden states and router logits, the output scattered back.
    hit = 8 * (1 - (6 / 8) ** rows)
    experts = grouped_gemm(20, 32, 32, hit, "bf16", "fp4")
    experts += grouped_gemm(20, 32, 16, hit, "bf16", "fp4")
    gather = 2e-6 + rows * (32 + 8) * 2 / 2 / LINK
    scatter = 2e-6 + rows * 32 * 2 / 2 / LINK
    assert _price([ep_off], step, spec)["roofline.moe"] == pytest.approx(
        _module(experts, spec) + gather + scatter
    )
    # A graph's pad rows all carry one token: 6 real tokens and the pad reach
    # the experts, every row still computed.
    padded = BatchView(step.requests, moe_rows=rows, moe_tokens=6)
    hit7 = activated_experts(8, 2, 7)
    experts = grouped_gemm(20, 32, 32, hit7, "bf16", "fp4")
    experts += grouped_gemm(20, 32, 16, hit7, "bf16", "fp4")
    assert _price([ep_off], padded, spec)["roofline.moe"] == pytest.approx(
        _module(experts, spec) + gather + scatter
    )
    # EP2: the busiest rank's share at the full intermediate, half the experts
    # read; each token sent once to each rank holding one of its two experts.
    load = max_rank_load(rows, 2, 8, 2)
    experts = grouped_gemm(load, 64, 32, hit / 2, "fp8", "fp4")
    experts += grouped_gemm(load, 32, 32, hit / 2, "fp8", "fp4")
    sent = tokens * 32 * 2 * (1 - comb(4, 2) / comb(8, 2))
    dispatch = 2e-6 + sent / 2 / LINK
    combine = 2e-6 + sent * 2 / 2 / LINK
    assert _price([ep2], step, spec)["roofline.moe"] == pytest.approx(
        _module(experts, spec) + dispatch + combine
    )
    # One data-parallel rank: its own rows, the output all-reduced.
    hit = activated_experts(8, 2, 3)
    experts = grouped_gemm(6, 32, 32, hit, "bf16", "fp4")
    experts += grouped_gemm(6, 32, 16, hit, "bf16", "fp4")
    experts += allreduce(2, tokens * 32 * 2)
    one = view((tokens, tokens, False))
    assert _price([one_rank], one, spec)["roofline.moe"] == pytest.approx(
        _module(experts, spec)
    )


@BOUND
def test_sparse_attention_and_the_hyper_connection_mix_by_hand(spec):
    # Window 4, every 3 tokens compressed, the indexer keeping the top 2.
    attn = Module("attn", "sparse_attention", (2, 8, 4, 3, 2, 3, 4), "fp8")
    mix = Module("mix", "mhc", (4, 16))
    rows = [(3, 10, False), (1, 7, True)]
    keys = scored = 0
    for q, kv, _ in rows:
        for p in range(kv - q + 1, kv + 1):
            keys += min(p, 4) + min(p // 3, 2)
            scored += p // 3
    # Each row reads its window and its top 2 compressed entries once, and
    # every compressed index key once.
    read = (4 + 2) + (4 + 2)
    work = Work({"bf16": 4 * 2 * 8 * keys}, 2 * 4 * 2 * 8 * 2 + read * 8)
    work += Work({"fp8": 2 * 3 * 4 * scored}, (3 + 2) * 4)
    # Read the residual's four copies and the output, write both back; the
    # 24 mixing weights' fp32 projection of the four copies.
    moved = 2 * 4 * 5 * 16 * 2 + 24 * 4 * 16 * 4
    flops = 2 * 4 * 24 * 4 * 16
    by_name = _price([attn, mix], view(*rows), spec)

    assert by_name["roofline.sparse_attention"] == pytest.approx(_module(work, spec))
    assert by_name["roofline.mhc"] == pytest.approx(
        _module(Work({"bf16": flops}, moved), spec)
    )


@BOUND
def test_a_split_step_lays_each_layers_micro_batches_on_two_streams(spec):
    layers = []
    for i in range(2):
        layers += [
            Module(f"layers.{i}.attn", "attention", (4, 1, 8), layer=i),
            Module(f"layers.{i}.proj", "gemm", (32, 32), layer=i),
            Module(f"layers.{i}.experts", "moe", EP_OFF, "fp4", ranks=2, layer=i),
        ]
    head = Module("head", "lm_head", (64, 32))
    # This rank's 6 tokens split 3 and 3, cutting the first request: its last
    # token attends the first three as history. The peer splits 5 and 5.
    step = view((4, 4, False), (2, 10, False), tbo=((3, 5), (3, 5)))
    halves = [([3], [3]), ([1, 2], [4, 10])]
    hit = activated_experts(8, 2, 8)
    experts = grouped_gemm(16, 32, 32, hit, "bf16", "fp4")
    experts += grouped_gemm(16, 32, 16, hit, "bf16", "fp4")
    segments = []
    for q, kv in halves:
        compute = _module(attention(q, kv, 4, 8, 1), spec)
        compute += _module(gemm(sum(q), 32, 32), spec)
        gather = bounds(allgather(2, 8 * 40 * 2), spec)["comm"]
        scatter = bounds(allgather(2, 8 * 32 * 2), spec)["comm"]
        segments.append([compute, gather, _module(experts, spec), scatter])
    layer = [list(pair) for pair in zip(*segments)]
    by_name = _price(layers + [head], step, spec)

    assert by_name["roofline.tbo"] == pytest.approx(overlap([layer, layer]))
    assert by_name["roofline.lm_head"] == _module(gemm(2, 64, 32), spec)
    assert set(by_name) == {"roofline.tbo", "roofline.lm_head"}


@pytest.mark.parametrize(
    "modules,step,refusal",
    [
        (
            [Module("experts", "moe", EP2[:-1] + (1,), "fp4", ranks=2, layer=0)],
            view((4, 4, False), tbo=((2, 2), (2, 2))),
            "split send and receive",
        ),
        (
            [Module("proj", "gemm", (32, 32), layer=0)],
            view((4, 4, False), tbo=((1, 3), (3, 1))),
            r"names no single pair of micro-batches for this rank's 4 tokens",
        ),
        (
            [Module("experts", "moe", EP_OFF, "fp4", ranks=2)],
            view((4, 4, False)),
            "a MoE over 2 data-parallel ranks, and no moe_rows",
        ),
    ],
    ids=["split-path", "ambiguous-split", "no-moe-rows"],
)
def test_a_step_with_no_price_is_refused_by_name(modules, step, refusal):
    with pytest.raises(CostRefused, match=refusal):
        StepRooflineBackend(machine(1e18), modules).price(step)


def test_the_batch_view_states_the_real_tokens_among_the_gathered_rows():
    # Two ranks decode 3 and 2 tokens in a graph of 4 rows each.
    r = runner(None, dp_size=2)
    r.drafter = None
    sync = SimpleNamespace(
        num_tokens_across_dp=torch.tensor([3, 2]), ub_tokens_across_dp=None
    )
    mode = SimpleNamespace(
        sync=sync,
        running_tokens_are_unified=True,
        running_tokens=4,
        use_cudagraph=True,
        running_bs=4,
    )
    view = projection.batch_view(decode(3), mode, r)

    assert (view.moe_rows, view.moe_tokens) == (8, 5)


BUILD = """
import json, pathlib, sys, torch
ep, root = sys.argv[1] == "1", pathlib.Path(sys.argv[2])
from atom.compass.capture import driverless
driverless.install(1, str(root))
from atom.config import Config, set_current_atom_config
config = Config(model=str(root / "model"), tensor_parallel_size=1, load_dummy="empty",
                enable_dp_attention=True, dp_logical_size=8, enable_expert_parallel=ep,
                kv_cache_dtype="fp8")
set_current_atom_config(config)
from atom.model_engine.model_runner import support_model_arch_dict
from atom.utils import resolve_obj_by_qualname
from atom.compass.runner.overrides import FactoriesOnMeta, process_weights_after_loading
from atom.compass.run.modules import walk
model = support_model_arch_dict[config.hf_config.architectures[0]]
torch.set_default_dtype(config.torch_dtype)
with FactoriesOnMeta():
    tree = resolve_obj_by_qualname(model)(config)
    process_weights_after_loading(tree)
found = [[m.kind, list(m.dims), m.fmt, m.act, m.layer] for m in walk(tree)]
print(json.dumps(found))
"""


@pytest.mark.parametrize("ep", [False, True], ids=["ep-off", "ep8"])
def test_the_v4_pro_dpa_tp8_build_classifies_every_module(ep, tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_bytes(CONFIG.read_bytes())
    # The declared device is an MI308X, whose EP-off default is a Triton MoE
    # layout the meta build cannot lay out; the MI355X deployment runs aiter's.
    env = {**os.environ, "PYTHONPATH": str(TREE), "ATOM_USE_TRITON_MOE": "0"}
    done = subprocess.run(
        [sys.executable, "-c", BUILD, str(int(ep)), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
        cwd=TREE,
        env=env,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    walked = json.loads(done.stdout.splitlines()[-1])
    found = collections.Counter(
        (kind, tuple(dims), fmt, act) for kind, dims, fmt, act, _ in walked
    )
    ratios = json.loads(CONFIG.read_text())["compress_ratios"][:61]

    assert not [m for m in found if m[0] == "unclassified"]
    assert {layer for *_, layer in walked} == set(range(61)) | {-1}
    assert found[("mhc", (4, 7168), "bf16", "bf16")] == 2 * 61
    for ratio, top in ((128, (0, 0, 0)), (4, (1024, 64, 128))):
        dims = (128, 512, 128, ratio) + top
        assert found[("sparse_attention", dims, "fp8", "bf16")] == ratios.count(ratio)
    moe = (384, 48, 6, 3072) if ep else (384, 384, 6, 384)
    assert found[("moe", moe + (7168, 8, 8, 0), "fp4", "bf16")] == 61
    # The fp8 linears read fp8 activations; the block's norms run in its mixes.
    assert found[("gemm", (65536, 1536), "fp8", "fp8")] == 61
    assert found[("act", (3072,), "bf16", "bf16")] == 61
    assert found[("norm", (7168,), "bf16", "bf16")] == 1
