# SPDX-License-Identifier: MIT
"""The graph-pool reservation a simulated run installs is the deployment's own.

`run.runner` installs the device readings on the worker; the runner's
`_estimate_cudagraph_overhead` then refuses any reading built for another
`enforce_eager` than its config's. Each case here installs through `run.runner`
with a Qwen3.5-27B config at TP1 on the suite's machine document, its memory
readings the entry `with_readings` publishes; the reservation cases read back
through that method.
"""

import json
from types import SimpleNamespace

import pytest
import torch
from test_memory_readings import (
    CAPTURE_SIZES,
    CONFIG_JSON,
    DOCUMENT,
    GPU_MEMORY_UTILIZATION,
    GRAPH_POOL_BY_DTYPE,
    MAX_NUM_BATCHED_TOKENS,
    META_8B,
    READINGS,
    WARMUP_TOKENS,
    overridden,
    with_readings,
)
from test_vertical_slice import _run_file
from transformers import PretrainedConfig

from atom.compass import run as compass_run
from atom.compass.artifacts import ArtifactRefusal
from atom.compass.runner.overrides import NonAllocatingRunner


def _installed(
    monkeypatch,
    tmp_path,
    *,
    eager,
    piecewise=False,
    dp=1,
    dpa=False,
    ep=False,
    sizes=None,
    hf_overrides=None,
    readings=READINGS[1],
    **run,
):
    """A worker's runner after `run.runner` installed its readings, the entry
    holding `readings`. With `hf_overrides`, the config.json leaves out each
    key they set."""
    monkeypatch.setenv(compass_run.ENV, str(_run_file(tmp_path, **run)))
    raw = json.loads(CONFIG_JSON.read_text())
    if hf_overrides is None:
        (tmp_path / "config.json").write_text(CONFIG_JSON.read_text())
        hf = PretrainedConfig.from_dict(raw["text_config"])
    else:
        text = {k: v for k, v in raw["text_config"].items() if k not in hf_overrides}
        hf, _ = overridden(tmp_path, raw | {"text_config": text}, hf_overrides)
    runner = SimpleNamespace(
        config=SimpleNamespace(
            model=str(tmp_path),
            hf_config=hf,
            hf_overrides=hf_overrides,
            tensor_parallel_size=1,
            enable_dp_attention=dpa,
            enable_expert_parallel=ep,
            parallel_config=SimpleNamespace(
                data_parallel_size=dp, data_parallel_rank=0
            ),
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            max_num_seqs=256,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            torch_dtype=torch.bfloat16,
            enforce_eager=eager,
            compilation_config=SimpleNamespace(
                cudagraph_capture_sizes=list(sizes or CAPTURE_SIZES)
            ),
            capture_sizes=list(sizes or CAPTURE_SIZES),
            kv_transfer_config={},
        ),
        _piecewise_cg_active=lambda: piecewise,
    )
    machine = run.get("machine", DOCUMENT)
    with_readings(runner, tmp_path / "store", readings, machine)
    compass_run.runner(runner, META_8B)
    return runner


def _reserved(monkeypatch, tmp_path, **kwargs):
    """What the runner reserves after `run.runner` installed its readings."""
    return NonAllocatingRunner._estimate_cudagraph_overhead(
        _installed(monkeypatch, tmp_path, **kwargs)
    )


def test_an_eager_deployment_reserves_nothing(monkeypatch, tmp_path):
    assert _reserved(monkeypatch, tmp_path, eager=True) == 0


def test_a_whole_graph_capture_reserves_a_fifth_of_the_activations(
    monkeypatch, tmp_path
):
    assert MAX_NUM_BATCHED_TOKENS == WARMUP_TOKENS
    activations = READINGS[1]["activation_bytes"]
    assert _reserved(monkeypatch, tmp_path, eager=False) == int(activations * 0.2)


def test_a_piecewise_capture_reserves_atoms_per_token_estimate(monkeypatch, tmp_path):
    reserved = _reserved(monkeypatch, tmp_path, eager=False, piecewise=True)
    assert reserved == GRAPH_POOL_BY_DTYPE["bfloat16"]


def test_the_piecewise_estimate_takes_the_data_parallel_width(monkeypatch, tmp_path):
    hf = PretrainedConfig.from_dict(json.loads(CONFIG_JSON.read_text())["text_config"])
    per_token = hf.hidden_size * 2 * hf.num_hidden_layers * 2.8 * 2**0.6
    reserved = _reserved(
        monkeypatch, tmp_path, eager=False, piecewise=True, dp=2, sizes=(1, 2)
    )
    assert reserved == int(per_token * 3)


def test_the_entry_s_activations_size_the_whole_graph_reservation(
    monkeypatch, tmp_path
):
    measured = dict(READINGS[1], activation_bytes=MAX_NUM_BATCHED_TOKENS * 180_480)
    reserved = _reserved(monkeypatch, tmp_path, eager=False, readings=measured)
    assert reserved == int(MAX_NUM_BATCHED_TOKENS * 180_480 * 0.2)


def test_a_field_only_hf_overrides_sets_reaches_the_installed_readings(
    monkeypatch, tmp_path
):
    runner = _installed(
        monkeypatch,
        tmp_path,
        eager=True,
        hf_overrides={"max_position_embeddings": 65536},
    )
    peak_torch = runner.compass_memory["readings"]["peak_torch"]
    (buffers,) = [row for row in peak_torch if row["name"] == "buffers"]
    assert buffers["source"].startswith("65536 positions")


@pytest.mark.parametrize(
    ("dpa", "dp", "ep", "label"),
    [(True, 8, False, "1x8"), (True, 8, True, "1x8x8"), (False, 2, False, 1)],
)
def test_each_layout_reads_its_own_entry_and_labels_its_readings_by_it(
    monkeypatch, tmp_path, dpa, dp, ep, label
):
    reserve = 8_190_000_000 + dp + ep
    measured = dict(READINGS[1], driver_and_collective_reserve_bytes=reserve)
    runner = _installed(
        monkeypatch, tmp_path, eager=True, dpa=dpa, dp=dp, ep=ep, readings=measured
    )
    (non_torch,) = runner.compass_memory["readings"]["non_torch"]
    assert non_torch["nbytes"] == reserve
    assert runner.compass_memory["label"].endswith(f" at TP{label}")


def test_an_entry_at_the_expert_parallel_off_layout_does_not_answer_one_on(
    monkeypatch, tmp_path
):
    _installed(monkeypatch, tmp_path, eager=True, dpa=True, dp=8)
    with pytest.raises(ArtifactRefusal, match="nearest entry differs in enable_exp"):
        _installed(
            monkeypatch, tmp_path, eager=True, dpa=True, dp=8, ep=True, readings=None
        )
