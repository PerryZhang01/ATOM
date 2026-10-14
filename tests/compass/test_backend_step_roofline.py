# SPDX-License-Identifier: MIT
"""The step-level fallback: a step the law cannot price is priced by the roofline.

The roofline's answer takes its method from the origins of the spec fields it
reads, and carries what the law declined. Steps
reach the backends through the runner's own `dummy_execution` and
`_group_step_seconds`, and the backends reach the runner through `run.runner`
from a run file's `step_linear` and `roofline_floor_s` keys.
"""

import copy
import json
import math
import types

import pytest
import test_run_graph_pool
from test_backend_step_linear import IN_HULL, _installed, backend, graphed, law, view
from test_memory_readings import CONFIG_JSON, DOCUMENT
from transformers import PretrainedConfig

from atom.compass.backends import (
    CostRefused,
    Granularity,
    KvGeometry,
    Method,
    ProvenanceMix,
    StepLinearBackend,
)
from atom.compass.backends.step_roofline import (
    FIELDS,
    StepFallbackBackend,
    StepRooflineBackend,
)
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
    return StepRooflineBackend(
        MACHINE,
        parameters=PARAMETERS,
        weight_bytes=WEIGHTS,
        kv_token_bytes=KV,
        host_floor_s=floor,
    )


def fallback():
    return StepFallbackBackend(backend(), roofline())


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
    (term,) = roofline(floor).price(batch).terms

    assert term.name == f"roofline.{bound}"
    assert term.seconds == seconds
    assert term.provenance.method is Method.DECLARED
    assert roofline().granularity is Granularity.STEP


@pytest.mark.parametrize("batch", IN_HULL, ids=range(len(IN_HULL)))
def test_a_step_the_law_prices_is_the_laws_answer(batch):
    assert fallback().price(batch) == backend().price(batch)
    assert fallback().granularity is Granularity.STEP


@pytest.mark.parametrize(
    "batch",
    [view((1, 10, True), rung=4), view((8, 8, False), (1, 10, True))],
    ids=["no-rung", "mixed"],
)
def test_a_step_the_law_refuses_is_priced_by_the_roofline_naming_what_declined(batch):
    with pytest.raises(CostRefused) as refused:
        backend().price(batch)
    cost = fallback().price(batch)
    mix = ProvenanceMix()
    mix.record(cost)

    assert cost.seconds == roofline().price(batch).seconds
    assert cost.refusals == refused.value.declined
    assert mix.seconds_by_method() == {(Method.DECLARED, False): cost.seconds}


def test_a_fallback_across_granularities_is_refused():
    class OpLevel(StepLinearBackend):
        @property
        def granularity(self):
            return Granularity.OP

    with pytest.raises(ValueError, match="inside one granularity"):
        StepFallbackBackend(OpLevel(backend().law), roofline())


def test_a_runner_whose_law_lacks_its_rung_prices_the_step_by_the_roofline():
    priced = graphed(fallback()).dummy_execution().predicted_s

    assert priced == roofline().price(view((1, 1, True), rung=1)).seconds


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
    installed = _installed(
        monkeypatch, tmp_path, step_linear=law(), roofline_floor_s=0.004
    )
    hf = PretrainedConfig.from_dict(json.loads(CONFIG_JSON.read_text())["text_config"])
    kv = KvGeometry.from_hf_config(hf, block_size=16, kv_dtype="bf16")

    assert isinstance(installed, StepFallbackBackend)
    assert isinstance(installed.first, StepLinearBackend)
    assert installed.roofline.host_floor_s == 0.004
    assert installed.roofline.parameters == 8_000_000_000
    assert installed.roofline.weight_bytes == 2 * 8_000_000_000
    assert installed.roofline.kv_token_bytes == kv.layers * kv.bytes_per_token_per_layer


def _spec(origin, origins=None):
    document = copy.deepcopy(DOCUMENT)
    document["provenance"]["origin"] = origin
    if origins is not None:
        document["provenance"]["origins"] = origins
    return MachineSpec.from_mapping(document)


@pytest.mark.parametrize(
    "spec,method,named",
    [
        (_spec("datasheet"), Method.ANALYTICAL, []),
        (_spec("probed"), Method.ANALYTICAL, []),
        (_spec("assumed"), Method.DECLARED, list(FIELDS)),
        (_spec("transferred-from:mi308x-hjbog18"), Method.DECLARED, list(FIELDS)),
        (
            _spec(
                "mixed",
                {p: {"origin": "datasheet"} for p in FIELDS}
                | {"device.memory.derate": {"origin": "assumed"}},
            ),
            Method.DECLARED,
            ["device.memory.derate"],
        ),
    ],
    ids=["datasheet", "probed", "assumed", "transferred", "one-derate-assumed"],
)
def test_the_roofline_is_declared_iff_a_spec_field_it_reads_is_not_its_own(
    spec, method, named
):
    backend = StepRooflineBackend(
        spec,
        parameters=PARAMETERS,
        weight_bytes=WEIGHTS,
        kv_token_bytes=KV,
        host_floor_s=0.0,
    )
    (term,) = backend.price(view((1, 900, True))).terms

    assert term.provenance.method is method
    assert [p for p in FIELDS if p in term.provenance.note] == named
    if named:
        assert "no successor" in term.provenance.note
