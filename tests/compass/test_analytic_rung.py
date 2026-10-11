# SPDX-License-Identifier: MIT
"""Tier 0 under tier a: a step the law cannot price is priced by the roofline.

The roofline's answer is `analytical` and carries what the law declined. Steps
reach the backends through the runner's own `dummy_execution` and
`_group_step_seconds`, and the backends reach the runner through `run.runner`
from a run file's `law` and `roofline_floor_s` keys.
"""

import json
import math
import types

import pytest
import test_run_graph_pool
from test_backend_coarse import IN_HULL, _installed, backend, graphed, law, view
from test_memory_readings import CONFIG_JSON, DOCUMENT
from transformers import PretrainedConfig

from atom.compass.backends import (
    CoarseBackend,
    CostRefused,
    KvGeometry,
    ProvenanceMix,
    Species,
    Tier,
)
from atom.compass.backends.analytic import Roofline, Tiered
from atom.compass.spec import MachineSpec

MACHINE = MachineSpec.from_mapping(DOCUMENT)
FLOPS = MACHINE.value("device.compute.bf16_flops") * MACHINE.value(
    "device.compute.derate"
)
BANDWIDTH = MACHINE.value("device.memory.bandwidth_bytes_per_s") * MACHINE.value(
    "device.memory.derate"
)
PARAMETERS, WEIGHTS, KV = 10**9, 2 * 10**9, 1000


def roofline(floor=0.0):
    return Roofline(
        MACHINE,
        parameters=PARAMETERS,
        weight_bytes=WEIGHTS,
        kv_token_bytes=KV,
        host_floor_s=floor,
    )


def tiered():
    return Tiered(backend(), roofline())


@pytest.mark.parametrize(
    "batch,floor,bound,seconds",
    [
        (view((1, 900, True)), 0.0, "memory", (WEIGHTS + KV * 901) / BANDWIDTH),
        (view((8192, 8192, False)), 0.0, "compute", 2 * PARAMETERS * 8192 / FLOPS),
        (view((1, 900, True)), 1.0, "host", 1.0),
    ],
    ids=["decode", "long-prefill", "floor"],
)
def test_a_step_is_the_largest_of_its_compute_memory_and_host_bounds(
    batch, floor, bound, seconds
):
    (term,) = roofline(floor).estimate(batch).terms

    assert term.name == f"roofline.{bound}"
    assert term.seconds == seconds
    assert term.provenance.species is Species.ANALYTICAL
    assert roofline().tier is Tier.ANALYTIC


@pytest.mark.parametrize("batch", IN_HULL, ids=range(len(IN_HULL)))
def test_a_step_the_law_prices_is_the_laws_answer(batch):
    assert tiered().estimate(batch) == backend().estimate(batch)
    assert tiered().tier is Tier.COARSE


@pytest.mark.parametrize(
    "batch",
    [view((1, 10, True), rung=4), view((8, 8, False), (1, 10, True))],
    ids=["no-rung", "mixed"],
)
def test_a_step_the_law_refuses_is_priced_analytical_naming_what_declined(batch):
    with pytest.raises(CostRefused) as refused:
        backend().estimate(batch)
    cost = tiered().estimate(batch)
    mix = ProvenanceMix()
    mix.record(cost)

    assert cost.seconds == roofline().estimate(batch).seconds
    assert cost.refusals == refused.value.declined
    assert mix.seconds_by_species() == {Species.ANALYTICAL: cost.seconds}


def test_a_runner_whose_law_lacks_its_rung_prices_the_step_analytical():
    priced = graphed(tiered()).dummy_execution().predicted_s

    assert priced == roofline().estimate(view((1, 1, True), rung=1)).seconds


@pytest.mark.parametrize("floor", [-1e-3, math.nan, math.inf])
def test_a_host_floor_that_is_not_a_duration_is_refused(floor):
    with pytest.raises(ValueError, match="host floor"):
        roofline(floor)


def _with_kv_fields(**fields):
    """`SimpleNamespace`, adding the KV fields to the runner's config."""
    if "hf_config" in fields:
        fields |= {"kv_cache_block_size": 16, "kv_cache_dtype": "bf16"}
    return types.SimpleNamespace(**fields)


def test_a_run_file_floor_puts_the_roofline_under_the_law(monkeypatch, tmp_path):
    monkeypatch.setattr(test_run_graph_pool, "SimpleNamespace", _with_kv_fields)
    installed = _installed(monkeypatch, tmp_path, law=law(), roofline_floor_s=0.004)
    hf = PretrainedConfig.from_dict(json.loads(CONFIG_JSON.read_text())["text_config"])
    kv = KvGeometry.from_hf_config(hf, block_size=16, kv_dtype="bf16")

    assert isinstance(installed, Tiered)
    assert isinstance(installed.first, CoarseBackend)
    assert installed.roofline.host_floor_s == 0.004
    assert installed.roofline.parameters == 8_000_000_000
    assert installed.roofline.weight_bytes == 2 * 8_000_000_000
    assert installed.roofline.kv_token_bytes == kv.layers * kv.bytes_per_token_per_layer
