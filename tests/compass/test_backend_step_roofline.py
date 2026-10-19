# SPDX-License-Identifier: MIT
"""The step-level fallback: a step the law cannot price is priced by the roofline.

The roofline's answer is declared when an input of the bound that won is stated,
naming each such input, and carries what the law declined. Steps
reach the backends through the runner's own `dummy_execution` and
`_group_step_seconds`, and the backends reach the runner through `run.runner`
from a run file's `step_linear` and `step_roofline` keys.
"""

import copy
import json
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
    StepFallbackBackend,
    StepRooflineBackend,
)
from atom.compass.spec import MachineSpec

MACHINE = MachineSpec.from_mapping(DOCUMENT)
FLOPS = MACHINE.value("device.compute.bf16_flops") * MACHINE.value(
    "device.compute.derate.bf16"
)
BANDWIDTH = MACHINE.value("device.memory.bandwidth_bytes_per_s") * MACHINE.value(
    "device.memory.derate"
)
PARAMETERS, WEIGHTS, KV = 10**9, 2 * 10**9, 1000


def roofline():
    return StepRooflineBackend(
        MACHINE, parameters=PARAMETERS, weight_bytes=WEIGHTS, kv_token_bytes=KV
    )


def fallback():
    return StepFallbackBackend(backend(), roofline())


@pytest.mark.parametrize(
    "batch,bound,seconds",
    [
        (view((1, 900, True)), "memory", (WEIGHTS + KV * 901) / BANDWIDTH),
        (view((8192, 8192, False)), "compute", 2 * PARAMETERS * 8192 / FLOPS),
    ],
    ids=["decode", "long-prefill"],
)
def test_a_step_is_the_larger_of_its_compute_and_memory_bounds(batch, bound, seconds):
    (term,) = roofline().price(batch).terms

    assert term.name == f"roofline.{bound}"
    assert term.seconds == seconds
    assert term.provenance.method is Method.DECLARED
    assert roofline().granularity is Granularity.STEP


def test_the_compute_bound_reads_the_bf16_derate_not_another_formats():
    document = copy.deepcopy(DOCUMENT)
    document["device"]["compute"]["derate"] = {"bf16": 0.70, "fp8": 0.35}
    machine = MachineSpec.from_mapping(document)
    backend = StepRooflineBackend(
        machine, parameters=PARAMETERS, weight_bytes=WEIGHTS, kv_token_bytes=KV
    )

    (term,) = backend.price(view((8192, 8192, False))).terms

    assert term.name == "roofline.compute"
    assert term.seconds == 2 * PARAMETERS * 8192 / (2.5e15 * 0.70)


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


def _with_kv_fields(**fields):
    """`SimpleNamespace`, adding the KV fields to the runner's config."""
    if "hf_config" in fields:
        fields |= {"kv_cache_block_size": 16, "kv_cache_dtype": "bf16"}
    return types.SimpleNamespace(**fields)


def test_a_run_file_step_roofline_puts_the_roofline_under_the_law(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(test_run_graph_pool, "SimpleNamespace", _with_kv_fields)
    installed = _installed(monkeypatch, tmp_path, step_linear=law(), step_roofline=True)
    hf = PretrainedConfig.from_dict(json.loads(CONFIG_JSON.read_text())["text_config"])
    kv = KvGeometry.from_hf_config(hf, block_size=16, kv_dtype="bf16")

    assert isinstance(installed, StepFallbackBackend)
    assert isinstance(installed.first, StepLinearBackend)
    assert installed.roofline.parameters == 8_000_000_000
    assert installed.roofline.weight_bytes == 2 * 8_000_000_000
    assert installed.roofline.kv_token_bytes == kv.layers * kv.bytes_per_token_per_layer


def test_a_run_file_step_roofline_false_leaves_the_law_alone(monkeypatch, tmp_path):
    installed = _installed(
        monkeypatch, tmp_path, step_linear=law(), step_roofline=False
    )

    assert isinstance(installed, StepLinearBackend)


@pytest.mark.parametrize("value", [0.004, 1, "true", None])
def test_a_run_file_step_roofline_that_is_not_a_bool_is_refused(
    monkeypatch, tmp_path, value
):
    with pytest.raises(TypeError, match=f"step_roofline is {value!r}"):
        _installed(monkeypatch, tmp_path, step_linear=law(), step_roofline=value)


def test_a_run_file_roofline_floor_is_refused_naming_step_roofline(
    monkeypatch, tmp_path
):
    with pytest.raises(
        ValueError, match=r"renamed: roofline_floor_s \(now step_roofline\)"
    ):
        _installed(monkeypatch, tmp_path, step_linear=law(), roofline_floor_s=0.004)


def _spec(origin, origins=None):
    document = copy.deepcopy(DOCUMENT)
    document["provenance"]["origin"] = origin
    if origins is not None:
        document["provenance"]["origins"] = origins
    return MachineSpec.from_mapping(document)


# The spec fields the roofline's constructor reads.
READS = (
    "device.compute.bf16_flops",
    "device.compute.derate.bf16",
    "device.memory.bandwidth_bytes_per_s",
    "device.memory.derate",
)


# Each bound's batch, the spec fields it reads and its stated inputs.
BOUNDS = {
    "compute": (view((8192, 8192, False)), "device.compute.", ["parameter_count"]),
    "memory": (view((1, 900, True)), "device.memory.", []),
}


@pytest.mark.parametrize("bound", BOUNDS)
@pytest.mark.parametrize(
    "spec,named",
    [
        (_spec("datasheet"), []),
        (_spec("probed"), []),
        (_spec("assumed"), list(READS)),
        *[
            (
                _spec(
                    "mixed",
                    {p: {"origin": "datasheet"} for p in READS}
                    | {path: {"origin": "assumed"}},
                ),
                [path],
            )
            for path in READS
        ],
    ],
    ids=["datasheet", "probed", "assumed"] + [f"{path}-assumed" for path in READS],
)
def test_a_roofline_term_is_declared_iff_an_input_its_bound_reads_is_stated(
    spec, named, bound
):
    batch, prefix, keys = BOUNDS[bound]
    backend = StepRooflineBackend(
        spec, parameters=PARAMETERS, weight_bytes=WEIGHTS, kv_token_bytes=KV
    )
    (term,) = backend.price(batch).terms
    note = term.provenance.note
    fields = [p for p in named if p.startswith(prefix)]

    assert term.name == f"roofline.{bound}"
    assert term.provenance.method is (
        Method.DECLARED if fields or keys else Method.ANALYTICAL
    )
    assert [k for k in ("parameter_count", "weights") if k in note] == keys
    assert [p for p in READS if p in note] == fields


def test_a_memory_bound_step_over_meta_build_weights_is_analytical(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(test_run_graph_pool, "SimpleNamespace", _with_kv_fields)
    datasheet = copy.deepcopy(DOCUMENT)
    datasheet["provenance"]["origin"] = "datasheet"
    installed = _installed(
        monkeypatch,
        tmp_path,
        step_linear=law(),
        step_roofline=True,
        machine=datasheet,
    )
    (term,) = installed.roofline.price(view((1, 90000, True))).terms
    note = term.provenance.note

    assert term.name == "roofline.memory"
    assert term.provenance.method is Method.ANALYTICAL
    assert "weights" not in note and "parameter_count" not in note


def test_a_compute_bound_step_over_parameters_off_the_build_is_analytical():
    """Without a stated parameter count the roofline counts the meta build's,
    so a compute bound on a datasheet spec states nothing."""
    backend = StepRooflineBackend(
        _spec("datasheet"),
        parameters=PARAMETERS,
        weight_bytes=WEIGHTS,
        kv_token_bytes=KV,
        parameters_stated=False,
    )
    (term,) = backend.price(BOUNDS["compute"][0]).terms

    assert term.name == "roofline.compute"
    assert term.provenance.method is Method.ANALYTICAL
    assert term.provenance.note == ""
