# SPDX-License-Identifier: MIT
"""Each memory term's method, site by site, from the origin of what it reads."""

import json
import pathlib

import pytest
from test_memory_readings import (
    CONFIG_JSON,
    model_terms,
    readings_at,
    spec_of,
)
from transformers import PretrainedConfig

from atom.compass.backends.provenance import Method
from atom.compass.memory import (
    MemoryRefusal,
    Term,
    footprint_terms,
    predicts,
    reserves,
)
from atom.compass.memory.check import parts
from atom.compass.run import _width_keys
from atom.compass.spec import MachineSpec

ROOT = pathlib.Path(__file__).resolve().parents[2]
HJBOG39 = ROOT / "scripts" / "compass" / "machines" / "mi308x-hjbog39.json"
SPEC_TERMS = ("capacity", "driver and collective reserve", "load residue", "persistent")


@pytest.fixture(scope="module")
def qwen():
    return PretrainedConfig.from_dict(
        json.loads(CONFIG_JSON.read_text())["text_config"]
    )


def _spec_terms(readings):
    every = {t.name: t for r in readings.as_dict().values() for t in r.terms}
    return {name: every[name] for name in SPEC_TERMS}


@pytest.mark.parametrize(
    "origin, method",
    [
        ("datasheet", Method.ANALYTICAL),
        ("probed", Method.MEASURED),
        ("assumed", Method.DECLARED),
        ("transferred-from:mi300x-8gpu", Method.DECLARED),
    ],
)
def test_a_spec_field_used_as_the_term_takes_its_method_from_its_origin(
    qwen, origin, method
):
    for tp_width in (1, 2):
        terms = _spec_terms(readings_at(spec_of(origin), qwen, tp_width))
        assert {t.method for t in terms.values()} == {method}, tp_width
        if method is Method.DECLARED:
            for term in terms.values():
                assert f"is {origin} in the machine spec" in term.note


def test_per_field_origins_of_a_real_machine_file_decide_each_term(qwen):
    spec = MachineSpec.from_mapping(_width_keys(json.loads(HJBOG39.read_text())))
    terms = _spec_terms(readings_at(spec, qwen, 1))
    assert {t.method for t in terms.values()} == {Method.MEASURED}
    # Its graph-pool line is assumed, so what it predicts is declared.
    pool = predicts(spec, tp_width=1, captured_tokens=1023)
    assert {t.method for t in pool.terms} == {Method.DECLARED}


@pytest.mark.parametrize(
    "origin, method",
    [("probed", Method.ESTIMATED), ("assumed", Method.DECLARED)],
)
@pytest.mark.parametrize("tp_width", [1, 2])
def test_the_pool_prediction_is_estimated_if_probed_and_declared_if_assumed(
    origin, method, tp_width
):
    pool = predicts(spec_of(origin), tp_width=tp_width, captured_tokens=1023)
    for term in pool.terms:
        assert term.method is method
        assert not term.validated


def test_the_whole_graph_reservation_combines_with_the_activations_term():
    declared = Term("activations", 1000, Method.DECLARED, "a formula", "a successor")
    estimated = Term("activations", 1000, Method.ESTIMATED, "an entry")
    (term,) = reserves(activations=declared).terms
    assert term.method is Method.DECLARED and "which is declared" in term.note
    (term,) = reserves(activations=estimated).terms
    assert term.method is Method.ESTIMATED and term.nbytes == 200


def test_the_piecewise_reservation_is_analytical(qwen):
    reading = readings_at(spec_of("assumed"), qwen, 1).cudagraph_overhead
    assert {t.method for t in reading.terms} == {Method.ANALYTICAL}


def test_meta_build_weights_are_analytical_and_the_entry_s_activations_measured(qwen):
    model = model_terms(qwen)
    assert model.weights.method is Method.ANALYTICAL
    assert model.activations.method is Method.MEASURED


def test_a_recording_is_measured():
    (term,) = footprint_terms({"peak_torch": 1}, source="a printout")
    assert term.method is Method.MEASURED


def test_only_an_estimated_term_can_be_validated():
    assert Term("a", 1, Method.ESTIMATED, "an entry", validated=True).validated
    with pytest.raises(ValueError, match="a: a measured answer cannot be validated"):
        Term("a", 1, Method.MEASURED, "a probe", validated=True)


def test_a_reading_as_one_term_combines_its_terms_methods(qwen):
    peak = readings_at(spec_of("probed"), qwen, 1).peak_torch
    term = peak.as_term()
    assert term.method is Method.DECLARED
    assert "It sums activations, which are declared" in term.note
    measured = readings_at(spec_of("probed"), qwen, 1).non_torch.as_term()
    assert measured.method is Method.MEASURED and not measured.note


def test_a_hand_in_recorded_by_basis_refuses_by_name(tmp_path):
    row = {"name": "weights", "nbytes": 1, "basis": "obtained", "source": "a"}
    hand_in = tmp_path / "lp-engine.json"
    hand_in.write_text(
        json.dumps(
            {
                "memory": {
                    "label": "an old run",
                    "tokens": 1,
                    "readings": {"peak_torch": [row]},
                    "predicts": None,
                }
            }
        )
    )
    with pytest.raises(MemoryRefusal, match="records each term's basis"):
        parts(hand_in, tmp_path / "server.log")
