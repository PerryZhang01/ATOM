# SPDX-License-Identifier: MIT
"""The five device readings, taken off a spec, with the device readings broken.

Every test here runs with `torch.cuda.mem_get_info` and
`torch.cuda.memory_stats` replaced by functions that raise. That is the point of
the fixture and not a precaution: the package claims that no reading comes off
a card, and a claim of that shape is worth what it costs to falsify.
It is checked twice over and the two checks fail differently -- the patch would
catch a call made at run time, and `test_the_package_imports_no_device` catches
a device import. It reads every import statement of every module, in every
branch, and it imports each module in a fresh interpreter, which sees a name
whose lookup loads the engine (`atom.LLMEngine`, `from atom import *`) and a
module reached through another. Neither sees a load reached only by code that
does not run at import (a function body that is not called, or a branch not
taken), whether it is an import statement or an attribute access, unless the
statement names a forbidden module itself: `importlib.import_module("torch")`,
`atom.LLMEngine` or `from atom import LLMEngine` inside a `def`, or a `def`
that imports a module which imports torch.

The model is the vendored Qwen3.8-27B config, as `test_backend_kv_geometry.py`
uses it, so the geometry here and the KV geometry there are the same model. The
spec is a complete MI355X document written out below rather than imported from
the spec tests, because the per-term table below is meant to be readable beside
its inputs.

Two things are asserted as byte counts rather than as properties, because what
this file pins is a per-term table and a table nobody checked is a claim. The
terms that come off the spec are asserted at their exact values, and so is the
clean-box identity that produces `free`.
"""

import ast
import copy
import json
import pathlib
import subprocess
import sys

import pytest
import torch
from transformers import PretrainedConfig

import atom.compass.memory as memory_package
from atom.compass.backends.geometry import dtype_bytes as element_bytes
from atom.compass.memory import (
    Basis,
    MemoryRefusal,
    ModelTerms,
    PiecewiseCapture,
    Reading,
    Term,
    capture_token_shapes,
    device_readings,
    piecewise_per_token_bytes,
    predicts,
    reserves,
)
from atom.compass.spec import MachineSpec, SpecRefusal
from atom.config import get_hf_config

# The package as the suite actually imported it, never as a walk up from this
# file: if `atom` ever resolves from another root -- a PYTHONPATH ahead of the
# tree, an installed copy, a staged snapshot -- a path-derived PACKAGE would
# read one tree while every other test here imports another, and pass.
PACKAGE = pathlib.Path(memory_package.__file__).parent
PACKAGE_DOTTED = memory_package.__name__
CONFIG_JSON = pathlib.Path(__file__).with_name("qwen3_5_27b_config.json")

#: The deployment's own numbers. They belong to ATOM's config, not to the spec,
#: which is why they are written here and not in the document below.
WARMUP_TOKENS = 8192
MAX_NUM_BATCHED_TOKENS = 8192
CAPTURE_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
GPU_MEMORY_UTILIZATION = 0.9
#: The model's stated parameter count. A meta build replaces it; see the note
#: the weights term carries.
PARAMETERS = 27_000_000_000

STACK = {"rocm": "7.2.4", "aiter": "f4e7c7509", "rccl": "2.22.3"}

TOKENIZER = {
    "id": "qwen3-151k-bpe",
    "backend": "fast",
    "vocab_size": 151936,
    "fingerprint": "sha256:" + "a" * 64,
    "applies_to": ["Qwen3ForCausalLM"],
    "encode_fixed_s": 3.0e-4,
    "encode_tokens_per_s": 2.0e6,
    "decode_fixed_s": 1.5e-4,
    "decode_tokens_per_s": 3.0e6,
    "derate": 0.85,
}

DOCUMENT = {
    "schema_version": 1,
    "name": "mi355x-8gpu-2node",
    "provenance": {
        "authored_by": "tests/compass/test_memory_readings.py",
        "date": "2026-09-22",
        "method": "declared",
    },
    "host": {
        "cpu": {"cores_physical": 96, "cores_logical": 192},
        "tokenizers": [TOKENIZER],
        "ipc": {"zmq_roundtrip_s": 5.0e-5, "shm_broadcast_s": 2.0e-5},
        "admission_fixed_s": 9.0e-3,
    },
    "device": {
        "name": "MI355X",
        "arch": "gfx950",
        "count_per_node": 8,
        "memory": {
            "capacity_bytes": 288.0e9,
            "bandwidth_bytes_per_s": 8.0e12,
            "derate": 0.85,
        },
        "compute": {"bf16_flops": 2.5e15, "fp8_flops": 5.0e15, "derate": 0.70},
        "runtime_constants": {
            "driver_and_collective_reserve_bytes": {
                1: 970.0e6,
                2: 7.2e9,
                4: 7.6e9,
                8: 11.2e9,
            },
            "allocator_retained_after_load_bytes": {
                1: 1.1e6,
                2: 2.17e9,
                4: 2.17e9,
                8: 2.17e9,
            },
            "persistent_forward_buffer_bytes": 124.0e6,
            "cudagraph_pool": {
                "w1_base_bytes": 95.5e6,
                "w1_bytes_per_captured_token": 0.318e6,
                "w_gt1_flat_bytes": 109.0e6,
            },
        },
        "software_pinned_to": dict(STACK),
    },
    "interconnect": {
        "intra_node": {
            "topology": "fully_connected",
            "link_bandwidth_bytes_per_s": 1.0e12,
            "link_latency_s": 2.0e-6,
            "derate": 0.80,
        },
        "inter_node": {
            "link_bandwidth_bytes_per_s": 5.0e10,
            "link_latency_s": 5.0e-6,
            "derate": 0.80,
        },
        "router_relay_s": 1.5e-3,
    },
}


@pytest.fixture(autouse=True)
def no_device_readings(monkeypatch):
    """Both device readings raise, for every test in this file."""

    def refuse(*args, **kwargs):
        raise AssertionError(
            "a device reading was taken; this package promises that none is"
        )

    monkeypatch.setattr(torch.cuda, "mem_get_info", refuse)
    monkeypatch.setattr(torch.cuda, "memory_stats", refuse)


@pytest.fixture(scope="module")
def spec():
    return MachineSpec.from_mapping(copy.deepcopy(DOCUMENT))


@pytest.fixture(scope="module")
def qwen():
    raw = json.loads(CONFIG_JSON.read_text())
    return PretrainedConfig.from_dict(raw["text_config"])


#: The terms sized by the model's element size, at TP1, for a 2-byte and a
#: 4-byte dtype. With bfloat16 alone, a size that collapsed to the literal 2
#: would read exactly like the right one; float32 is every term doubled.
MODEL_TERMS_BY_DTYPE = {
    "bfloat16": {
        "weights": 54_000_000_000,
        "buffers": 33_554_432,
        "activations": 805_306_368,
    },
    "float32": {
        "weights": 108_000_000_000,
        "buffers": 67_108_864,
        "activations": 1_610_612_736,
    },
}
GRAPH_POOL_BY_DTYPE = {"bfloat16": 1_877_213_184, "float32": 3_754_426_368}


@pytest.fixture(scope="module", params=sorted(MODEL_TERMS_BY_DTYPE))
def resident(request, qwen):
    """The same model with its tensors resident at another dtype."""
    config = copy.deepcopy(qwen)
    config.dtype = getattr(torch, request.param)
    return request.param, config


def ladder():
    """The captured shapes, as ATOM's capture loop would take them."""
    return capture_token_shapes(
        CAPTURE_SIZES, max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS
    )


def reserved(qwen, total_bytes):
    """What ATOM's own estimator would reserve for this ladder.

    The element size comes off the config, the way the module under test takes
    it. Writing 2 here would be the same number and the same defaulting the
    source just stopped doing -- a test that hardcodes what its subject derives
    stops being able to see the derivation break.
    """
    return reserves(
        piecewise=PiecewiseCapture(
            per_token_bytes=piecewise_per_token_bytes(
                hidden_size=qwen.hidden_size,
                layers=qwen.num_hidden_layers,
                dtype_bytes=element_bytes(qwen.dtype),
            ),
            token_shapes=ladder(),
            budget_bytes=int(total_bytes * GPU_MEMORY_UTILIZATION),
        )
    )


def readings_at(spec, qwen, tp_width):
    total_bytes = int(spec.value("device.memory.capacity_bytes"))
    return device_readings(
        spec,
        tp_width=tp_width,
        model=ModelTerms.from_declared_config(
            qwen,
            parameter_count=PARAMETERS,
            tp_size=tp_width,
            warmup_tokens=WARMUP_TOKENS,
        ),
        cudagraph_overhead=reserved(qwen, total_bytes),
    )


# --- the per-term table ------------------------------------------------------

#: Every term of every reading at TP1 and TP2, as bytes. It is written down so
#: that a change to any one term is a change to this table rather than to a
#: total that could absorb it. All five readings are here, `cudagraph_overhead`
#: included: the ratio band in
#: `test_the_two_graph_pool_numbers_disagree_by_the_recorded_factor` lets a 7%
#: move in LIVE_TENSORS_PER_LAYER through at width 1.
EXPECTED = {
    1: {
        "total": {"capacity": 288_000_000_000},
        "peak_torch": {
            "weights": 54_000_000_000,
            "buffers": 33_554_432,
            "load residue": 1_100_000,
            "persistent": 124_000_000,
            "activations": 805_306_368,
        },
        "non_torch": {"driver and collective reserve": 970_000_000},
        "cudagraph_overhead": {"per-token x captured tokens": 1_877_213_184},
        "free": {
            "total": 288_000_000_000,
            "less peak_torch": -54_963_960_800,
            "less non_torch": -970_000_000,
        },
    },
    2: {
        "total": {"capacity": 288_000_000_000},
        "peak_torch": {
            "weights": 27_000_000_000,
            "buffers": 33_554_432,
            "load residue": 2_170_000_000,
            "persistent": 124_000_000,
            "activations": 805_306_368,
        },
        "non_torch": {"driver and collective reserve": 7_200_000_000},
        "cudagraph_overhead": {"per-token x captured tokens": 1_877_213_184},
        "free": {
            "total": 288_000_000_000,
            "less peak_torch": -30_132_860_800,
            "less non_torch": -7_200_000_000,
        },
    },
}


@pytest.mark.parametrize("tp_width", [1, 2])
def test_the_five_readings_are_a_per_term_table(spec, qwen, tp_width, capsys):
    readings = readings_at(spec, qwen, tp_width)
    with capsys.disabled():
        print()
        print(readings.table())
    for name, terms in EXPECTED[tp_width].items():
        reading = readings.as_dict()[name]
        assert {t.name: t.nbytes for t in reading.terms} == terms, name


@pytest.mark.parametrize("tp_width", [1, 2])
def test_free_is_a_clean_box_and_not_a_reading(spec, qwen, tp_width):
    # Derived, so `(total - free)` can never carry a neighbour's bytes.
    readings = readings_at(spec, qwen, tp_width)
    assert readings.free.total == (
        readings.total.total - readings.peak_torch.total - readings.non_torch.total
    )
    assert all(term.basis is Basis.DERIVED for term in readings.free.terms)


@pytest.mark.parametrize("tp_width", [1, 2])
def test_every_term_names_where_it_came_from(spec, qwen, tp_width):
    readings = readings_at(spec, qwen, tp_width)
    for name, reading in readings.as_dict().items():
        for term in reading.terms:
            assert term.source.strip(), f"{name}.{term.name}"
            if term.basis is Basis.SPEC:
                path = term.source.partition("[")[0].partition(" x ")[0]
                assert path in spec.values, f"{name}.{term.name} names {path!r}"
            if term.basis is Basis.DECLARED:
                assert term.note.strip(), f"{name}.{term.name}"


def test_the_declared_terms_are_exactly_weights_buffers_and_activations(spec, qwen):
    # Each is a declared formula until its source exists: a meta build for
    # weights, a recording for buffers, a liveness walk for activations.
    readings = readings_at(spec, qwen, 1)
    assert set(readings.peak_torch.declared) == {"weights", "buffers", "activations"}


@pytest.mark.parametrize("tp_width", [1, 2])
def test_the_min_budget_free_clamp_cannot_bind(spec, qwen, tp_width):
    # `test_kv_budget_engine.py` checks with the engine that the clamp does not
    # bind. What is checked here is the only thing checkable without the engine:
    # with `free` a clean box, the budget branch is below it at every
    # utilisation the engine accepts.
    readings = readings_at(spec, qwen, tp_width)
    total = readings.total.total
    for utilisation in (0.5, 0.7, 0.9, 0.95, 1.0):
        non_kv = (
            readings.peak_torch.total
            + readings.non_torch.total
            + readings.cudagraph_overhead.total
            + int(total * 0.02)
        )
        available = int(total * utilisation) - non_kv
        assert min(available, readings.free.total) == available, utilisation


# --- a width nobody measured -------------------------------------------------


def test_an_unmeasured_width_refuses_naming_the_field_and_the_width(spec, qwen):
    with pytest.raises(SpecRefusal) as refusal:
        readings_at(spec, qwen, 3)
    message = str(refusal.value)
    assert "driver_and_collective_reserve_bytes" in message
    assert "width 3" in message
    assert "[1, 2, 4, 8]" in message


def test_the_refusal_names_whichever_table_is_short(qwen):
    # The two width tables are read separately, so a width present in one and
    # missing from the other refuses on the one that is missing it, by name.
    document = copy.deepcopy(DOCUMENT)
    del document["device"]["runtime_constants"]["allocator_retained_after_load_bytes"][
        2
    ]
    short = MachineSpec.from_mapping(document)
    with pytest.raises(SpecRefusal) as refusal:
        readings_at(short, qwen, 2)
    message = str(refusal.value)
    assert "allocator_retained_after_load_bytes" in message
    assert "width 2" in message
    assert "nothing here to interpolate along" in message


def test_a_configuration_that_does_not_fit_refuses_rather_than_clamping(spec, qwen):
    with pytest.raises(MemoryRefusal) as refusal:
        device_readings(
            spec,
            tp_width=1,
            model=ModelTerms.from_declared_config(
                qwen,
                parameter_count=400_000_000_000,
                tp_size=1,
                warmup_tokens=WARMUP_TOKENS,
            ),
            cudagraph_overhead=reserved(qwen, 288.0e9),
        )
    assert "clean box is negative" in str(refusal.value)


# --- the two graph-pool functions --------------------------------------------


def test_the_predicting_function_cannot_be_spent_as_the_reserving_one(spec, qwen):
    # Only the mirror of ATOM's estimator reserves anything; this is where
    # that is enforced rather than remembered.
    with pytest.raises(MemoryRefusal) as refusal:
        device_readings(
            spec,
            tp_width=1,
            model=ModelTerms.from_declared_config(
                qwen,
                parameter_count=PARAMETERS,
                tp_size=1,
                warmup_tokens=WARMUP_TOKENS,
            ),
            cudagraph_overhead=predicts(
                spec, tp_width=1, captured_tokens=sum(ladder())
            ),
        )
    assert "4-19x" in str(refusal.value)


@pytest.mark.parametrize("tp_width,low,high", [(1, 4.0, 5.0), (2, 17.0, 18.0)])
def test_the_two_graph_pool_numbers_disagree_by_the_recorded_factor(
    spec, qwen, tp_width, low, high
):
    # The two disagree by 4-19x across the recorded ladders. On this ladder
    # and this spec the measured pool is 4.46x smaller than ATOM's reservation
    # at width 1 and 17.2x at width 2, which is inside that band and is why
    # substituting one for the other is refused.
    reserving = reserved(qwen, 288.0e9).total
    predicted = predicts(spec, tp_width=tp_width, captured_tokens=sum(ladder())).total
    assert low < reserving / predicted < high


def test_the_predicted_pool_is_flat_above_width_one(spec):
    flat = {
        predicts(spec, tp_width=width, captured_tokens=tokens).total
        for width in (2, 4, 8)
        for tokens in (512, 1023, 4096)
    }
    assert flat == {109_000_000}


def test_a_ladder_drops_what_the_token_budget_cannot_schedule():
    assert capture_token_shapes(
        (1, 256, 512), q_buckets=(1, 4), max_num_batched_tokens=1024
    ) == (1, 4, 256, 512, 1024)


def test_the_reserving_branch_must_be_stated():
    with pytest.raises(ValueError, match="state one branch"):
        reserves()


def test_enforce_eager_reserves_nothing_and_says_why():
    reading = reserves(enforce_eager=True)
    assert reading.total == 0
    assert "enforce_eager" in reading.terms[0].source


# --- a reading is its terms --------------------------------------------------


def test_a_reading_cannot_be_spent_as_a_number():
    # A summed check read +13.8% while holding a 25% error in one term.
    reading = Reading("x", (Term("a", 3, Basis.DERIVED, "somewhere"),))
    assert not hasattr(reading, "__int__")
    assert not hasattr(reading, "__index__")
    assert "a" in str(reading) and "somewhere" in str(reading)


def test_a_term_without_a_source_is_refused():
    with pytest.raises(ValueError, match="states no source"):
        Term("a", 3, Basis.SPEC, "  ")


def test_a_declared_term_must_say_what_replaces_it():
    with pytest.raises(ValueError, match="does not say what replaces it"):
        Term("a", 3, Basis.DECLARED, "a coefficient")


def test_a_reading_with_no_terms_is_refused():
    with pytest.raises(ValueError, match="has no terms"):
        Reading("x", ())


def test_the_total_follows_the_terms_rather_than_being_stored():
    terms = (
        Term("a", 10, Basis.SPEC, "device.memory.capacity_bytes"),
        Term("b", -4, Basis.DERIVED, "a reading"),
    )
    assert Reading("x", terms).total == 6


# --- the assumptions this module makes, shown rather than held ---------------


def test_an_absent_partial_rotary_factor_says_so_in_the_table(qwen):
    # The one field whose absence produced the recorded 4x. 1.0 is the
    # right reading for a full-rotary model, so it is not refused -- but the
    # row must not look the same as a config that states 1.0.
    full = copy.deepcopy(qwen)
    del full.partial_rotary_factor
    stated = ModelTerms.from_declared_config(
        qwen, parameter_count=PARAMETERS, tp_size=1, warmup_tokens=WARMUP_TOKENS
    )
    assumed = ModelTerms.from_declared_config(
        full, parameter_count=PARAMETERS, tp_size=1, warmup_tokens=WARMUP_TOKENS
    )
    assert "absent from config, assumed" in assumed.buffers.source
    assert "absent from config, assumed" not in stated.buffers.source
    assert assumed.buffers.nbytes == 4 * stated.buffers.nbytes


def test_the_model_dtype_sizes_the_model_terms(qwen):
    # Read off the config, not a module constant: a term sized in fp32 where
    # the tensors are resident in bf16 is twice what the model holds.
    assert str(qwen.dtype).endswith("bfloat16")
    terms = ModelTerms.from_declared_config(
        qwen, parameter_count=PARAMETERS, tp_size=1, warmup_tokens=WARMUP_TOKENS
    )
    assert terms.buffers.nbytes == 262_144 * 64 * 2
    assert " x 2 B" in terms.buffers.source


def test_the_model_terms_follow_the_element_size(spec, resident):
    name, config = resident
    peak = readings_at(spec, config, 1).peak_torch
    expected = MODEL_TERMS_BY_DTYPE[name]
    assert {t.name: t.nbytes for t in peak.terms if t.name in expected} == expected


def test_the_graph_pool_reservation_follows_the_element_size(resident):
    name, config = resident
    assert reserved(config, 288.0e9).total == GRAPH_POOL_BY_DTYPE[name]


#: A `device.activations` entry: one real run's `peak - current` after warmup
#: over its warmup tokens, for this model at TP1.
ACTIVATIONS = {
    "id": "qwen3.8-27b",
    "fingerprint": "sha256:" + "b" * 64,
    "applies_to": ["Qwen3_5ForConditionalGeneration"],
    "bytes_per_token": {1: 180_480},
}


def test_a_measured_entry_prices_the_activations_only_for_its_model_and_width(qwen):
    document = copy.deepcopy(DOCUMENT)
    document["device"]["activations"] = [ACTIVATIONS]
    measured_spec = MachineSpec.from_mapping(document)
    arch, digest = ACTIVATIONS["applies_to"][0], ACTIVATIONS["fingerprint"]

    def activations(architecture, fingerprint, tp_width):
        return ModelTerms.from_declared_config(
            qwen,
            parameter_count=PARAMETERS,
            tp_size=tp_width,
            warmup_tokens=WARMUP_TOKENS,
            measured_activations=measured_spec.activations_for(
                architecture, fingerprint, tp_width
            ),
        ).activations

    term = activations(arch, digest, 1)
    assert term.basis is Basis.SPEC
    assert term.nbytes == WARMUP_TOKENS * 180_480
    assert "device.activations[qwen3.8-27b].bytes_per_token[1]" in term.source
    # Another size of the same architecture, an unmeasured width, and another
    # architecture keep the declared form.
    for key in (
        (arch, "sha256:" + "c" * 64, 1),
        (arch, digest, 2),
        ("Qwen3ForCausalLM", digest, 1),
    ):
        fallback = activations(*key)
        assert fallback.basis is Basis.DECLARED, key
        assert fallback.nbytes == MODEL_TERMS_BY_DTYPE["bfloat16"]["activations"]


def test_two_activation_entries_for_one_model_are_refused_where_the_spec_is_read():
    document = copy.deepcopy(DOCUMENT)
    document["device"]["activations"] = [ACTIVATIONS, dict(ACTIVATIONS, id="again")]
    with pytest.raises(SpecRefusal, match="resolves to two entries"):
        MachineSpec.from_mapping(document)


def test_a_config_with_no_dtype_refuses_rather_than_assuming_one(qwen):
    nameless = copy.deepcopy(qwen)
    del nameless.dtype
    with pytest.raises(MemoryRefusal, match="states no `dtype`, and every byte"):
        ModelTerms.from_declared_config(
            nameless,
            parameter_count=PARAMETERS,
            tp_size=1,
            warmup_tokens=WARMUP_TOKENS,
        )


@pytest.mark.parametrize(
    "field", ["hidden_size", "intermediate_size", "head_dim", "max_position_embeddings"]
)
def test_a_config_missing_a_geometry_field_refuses_naming_that_field(qwen, field):
    shapeless = copy.deepcopy(qwen)
    delattr(shapeless, field)
    with pytest.raises(MemoryRefusal) as refusal:
        ModelTerms.from_declared_config(
            shapeless,
            parameter_count=PARAMETERS,
            tp_size=1,
            warmup_tokens=WARMUP_TOKENS,
        )
    assert refusal.value.what == (
        f"this config states no `{field}`, and the memory model reads it"
    )
    assert "through ATOM's own config classes" in refusal.value.remedy


def test_the_negative_box_refusal_carries_its_decomposition(spec, qwen):
    # A refusal carries its decomposition too: the reader has to see which of the
    # six terms does not fit, and three of them are declared coefficients.
    with pytest.raises(MemoryRefusal) as refusal:
        device_readings(
            spec,
            tp_width=1,
            model=ModelTerms.from_declared_config(
                qwen,
                parameter_count=400_000_000_000,
                tp_size=1,
                warmup_tokens=WARMUP_TOKENS,
            ),
            cudagraph_overhead=reserved(qwen, 288.0e9),
        )
    message = str(refusal.value)
    for term in ("weights", "buffers", "load residue", "persistent", "activations"):
        assert term in message, term
    assert "gpu-memory-utilization" in message


def test_a_deployment_flag_is_not_labelled_geometry():
    # enforce_eager is a serving knob, so it belongs to the deployment rather
    # than to the machine or the model.
    assert reserves(enforce_eager=True).terms[0].basis is Basis.DEPLOYMENT
    assert not hasattr(Basis, "GEOMETRY")


# --- fields the model's config.json does not state ---------------------------

#: DeepSeek-V4-Pro's config.json: the fields the model terms read, and the
#: model type that routes it through `DeepseekV3Config`. It states no
#: `intermediate_size`, which that class defaults to 18432.
V4_PRO = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "model_type": "deepseek_v4",
    "hidden_size": 7168,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "max_position_embeddings": 1048576,
    "moe_intermediate_size": 3072,
    "num_experts_per_tok": 6,
    "n_shared_experts": 1,
    "n_routed_experts": 384,
    "num_hidden_layers": 61,
    "torch_dtype": "bfloat16",
}


def loaded(tmp_path, raw):
    """`raw` written as a model's config.json and loaded as `run.runner` sees
    it: ATOM's own config, and the file it came from."""
    (tmp_path / "config.json").write_text(json.dumps(raw))
    return (
        get_hf_config(str(tmp_path)),
        PretrainedConfig.get_config_dict(str(tmp_path))[0],
    )


def terms_of(config, config_json):
    return ModelTerms.from_declared_config(
        config,
        parameter_count=PARAMETERS,
        tp_size=1,
        warmup_tokens=WARMUP_TOKENS,
        config_json=config_json,
    )


def test_v4_pro_terms_name_only_fields_its_config_json_states(tmp_path):
    config, config_json = loaded(tmp_path, V4_PRO)
    assert type(config).__name__ == "DeepseekV3Config"
    assert config.intermediate_size == 18432  # the class default, still there
    terms = terms_of(config, config_json)
    assert terms.buffers.nbytes == 1_048_576 * 64 * 2
    assert terms.buffers.source == "1048576 positions x 64 qk_rope_head_dim x 2 B"
    assert terms.activations.nbytes == int(
        WARMUP_TOKENS * 2 * (2.8 * 7168 + 2 * (6 + 1) * 3072)
    )
    assert terms.activations.source == (
        "8192 warmup tokens x 2 B x (2.8 x 7168 hidden + 2 x (6 num_experts_per_tok"
        " + 1 n_shared_experts) x 3072 moe_intermediate_size)"
    )


@pytest.mark.parametrize(
    "dropped,named",
    [
        ("hidden_size", "hidden_size"),
        ("max_position_embeddings", "max_position_embeddings"),
        ("num_experts_per_tok", "num_experts_per_tok"),
        ("n_shared_experts", "n_shared_experts"),
        # Not an MoE model any more, so the dense field is read: unstated too.
        ("moe_intermediate_size", "intermediate_size"),
    ],
)
def test_a_field_only_the_config_class_supplies_is_refused_by_name(
    tmp_path, dropped, named
):
    config, config_json = loaded(
        tmp_path, {k: v for k, v in V4_PRO.items() if k != dropped}
    )
    assert getattr(config, named) is not None  # the class filled it in
    with pytest.raises(MemoryRefusal) as refusal:
        terms_of(config, config_json)
    assert refusal.value.what == (
        f"this config states no `{named}`, and the memory model reads it"
    )


def test_without_its_config_json_a_class_default_cannot_count_as_stated(tmp_path):
    config, _ = loaded(tmp_path, V4_PRO)
    with pytest.raises(MemoryRefusal, match="states no `hidden_size`"):
        terms_of(config, None)


#: Qwen3-30B-A3B's config.json, the fields the model terms read. The
#: architecture has no shared expert, so neither the file nor `Qwen3MoeConfig`
#: names one.
QWEN3_30B_A3B = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "model_type": "qwen3_moe",
    "hidden_size": 2048,
    "head_dim": 128,
    "intermediate_size": 6144,
    "max_position_embeddings": 40960,
    "moe_intermediate_size": 768,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 48,
    "torch_dtype": "bfloat16",
}


def test_an_moe_class_with_no_shared_expert_field_reads_zero(tmp_path):
    config, config_json = loaded(tmp_path, QWEN3_30B_A3B)
    assert type(config).__name__ == "Qwen3MoeConfig"
    terms = terms_of(config, config_json)
    # 8 x 768 is the dense 6144: the expert form changes no number here.
    assert terms.activations.nbytes == int(WARMUP_TOKENS * 2 * (2.8 * 2048 + 2 * 6144))
    assert terms.activations.source == (
        "8192 warmup tokens x 2 B x (2.8 x 2048 hidden + 2 x (8 num_experts_per_tok"
        " + 0 n_shared_experts (Qwen3MoeConfig has none)) x 768 moe_intermediate_size)"
    )


def test_a_bare_config_with_no_shared_expert_count_is_refused():
    raw = {k: v for k, v in QWEN3_30B_A3B.items() if k != "model_type"}
    config = PretrainedConfig.from_dict({**raw, "dtype": "bfloat16"})
    with pytest.raises(MemoryRefusal, match="states no `n_shared_experts`"):
        terms_of(config, raw)


@pytest.mark.parametrize(
    "extra,klass,named",
    [
        # A class that sizes its shared expert by width, not by count.
        (
            {"model_type": "qwen2_moe"},
            "Qwen2MoeConfig",
            "shared_expert_intermediate_size",
        ),
        # A class that counts its shared experts under another name.
        ({"model_type": "exaone_moe"}, "ExaoneMoeConfig", "num_shared_experts"),
        # A class with no shared-expert field, and a config.json that names one.
        (
            {"shared_expert_intermediate_size": 768},
            "Qwen3MoeConfig",
            "shared_expert_intermediate_size",
        ),
    ],
)
def test_a_config_naming_a_shared_expert_otherwise_is_refused(
    tmp_path, extra, klass, named
):
    config, config_json = loaded(tmp_path, {**QWEN3_30B_A3B, **extra})
    assert type(config).__name__ == klass
    with pytest.raises(MemoryRefusal, match="states no `n_shared_experts`") as refusal:
        terms_of(config, config_json)
    assert named in refusal.value.remedy


def test_a_dense_model_through_atoms_loader_keeps_its_terms(tmp_path, qwen):
    # Qwen3.8-27B loads as a text config class that has a default for every
    # field read here; its config.json states them all, under `text_config`.
    config, config_json = loaded(tmp_path, json.loads(CONFIG_JSON.read_text()))
    assert type(config).__name__ != "PretrainedConfig"
    assert terms_of(config, config_json) == terms_of(qwen, None)


# --- no device, structurally -------------------------------------------------

#: Roots the package may not import at all, and the prefixes of ATOM that reach
#: a driver. Device capability is configured here, never read off a device.
FORBIDDEN_ROOTS = frozenset({"torch", "transformers"})
FORBIDDEN_PREFIXES = ("atom.model_engine", "atom.model_ops", "atom.models")


def _forbidden(mod):
    return mod.split(".")[0] in FORBIDDEN_ROOTS or mod.startswith(FORBIDDEN_PREFIXES)


def _imported_names(tree):
    """Every module name a tree imports, with relative imports resolved.

    Resolving the level is what makes this test hold. `from ...model_engine
    import model_runner` carries `node.module == 'model_engine'` and
    `node.level == 3`, which matches neither a forbidden root nor a forbidden
    prefix; its two rows in the positive control,
    `test_the_import_guard_catches_what_it_claims_to`, fail when the level is
    ignored. `from ... import model_engine` carries `node.module is None`,
    which a truthiness guard skips entirely; here each imported name is
    resolved against the package instead, to `atom.model_engine`, and its row
    fails when that guard is put back. The absolute `from atom import
    model_engine` adds `atom.model_engine` beside `atom`, and its row fails
    without that.
    """
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = PACKAGE_DOTTED.split(".")[: -node.level + 1 or None]
                if node.module:
                    names.add(".".join(parts + [node.module]))
                else:
                    names.update(".".join(parts + [alias.name]) for alias in node.names)
            elif node.module:
                names.add(node.module)
                names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _device_imports(source):
    """The forbidden modules imported by running `source` inside the package, in
    a fresh interpreter (this one imported torch above) started in the root this
    suite imported `atom` from. The first forbidden import is refused before it
    loads, so a caught row costs an interpreter start, not an engine."""
    probe = (
        "import os, sys\n"
        "class Refuse:\n"
        "    def find_spec(name, *_):\n"
        f"        if name.split('.')[0] in {sorted(FORBIDDEN_ROOTS)} or name.startswith({FORBIDDEN_PREFIXES}):\n"
        "            os.write(1, b'\\n' + name.encode()), os._exit(0)\n"
        "sys.meta_path.insert(0, Refuse)\n"
        f"exec({source!r}, {{'__package__': {PACKAGE_DOTTED!r}}})\n"
        "print(*sys.modules)"
    )
    run = [sys.executable, "-c", probe]
    loaded = subprocess.check_output(run, cwd=PACKAGE.parents[2], text=True).split()
    return [name for name in loaded if _forbidden(name)]


@pytest.mark.parametrize("module", sorted(p.name for p in PACKAGE.glob("*.py")))
def test_the_package_imports_no_device(module):
    # The patched fixture catches a call; this catches the possibility of one.
    # The walk reads branches that never execute but only one level deep; the
    # import reads the whole closure but only what runs at import time.
    for name in sorted(_imported_names(ast.parse((PACKAGE / module).read_text()))):
        assert not _forbidden(name), f"{module} -> {name}"
    dotted = f"{PACKAGE_DOTTED}.{module[:-3]}".removesuffix(".__init__")
    assert _device_imports(f"import {dotted}") == [], module


@pytest.mark.parametrize(
    "source,caught",
    [
        ("import torch", True),
        ("import torch.cuda", True),
        ("from atom.model_engine.model_runner import ModelRunner", True),
        ("from ...model_engine import model_runner", True),
        ("from ...model_engine.model_runner import ModelRunner", True),
        ("from ... import model_engine", True),
        ("from atom import model_engine", True),
        ("from . import terms", False),
        ("from atom.compass.spec import MachineSpec", False),
    ],
)
def test_the_import_guard_catches_what_it_claims_to(source, caught):
    # A guard with no positive control is a guard nobody has seen work.
    names = _imported_names(ast.parse(source))
    assert any(map(_forbidden, names)) is caught, names


@pytest.mark.parametrize(
    "source,caught",
    [
        ("from ... import LLMEngine", True),
        ("import atom\natom.LLMEngine", True),
        ("from ... import *", True),
        ("from atom.compass.runner import overrides", True),
        ("import sys\nsys.stdout.write('x')\nfrom ... import LLMEngine", True),
        ("import io, sys\nsys.stdout = io.StringIO()\nfrom ... import LLMEngine", True),
        ("from . import terms", False),
    ],
)
def test_the_import_run_catches_what_the_walk_does_not(source, caught):
    # Every row passes the walk. Each True row reaches a device import anyway: a
    # name, not a module, loads the engine through `atom/__init__.py`'s lazy lookup,
    # or a module outside the package imports torch. The refusal goes to file
    # descriptor 1, so the two rows that write to or replace `sys.stdout` do not
    # hide it.
    assert not any(map(_forbidden, _imported_names(ast.parse(source))))
    assert bool(_device_imports(source)) is caught
