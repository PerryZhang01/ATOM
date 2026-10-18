# SPDX-License-Identifier: MIT
"""The five device readings, taken off a spec and a `memory_readings` entry,
with the device readings broken.

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
its inputs; `READINGS` is one rank's member of the entry at each width, and
`publish_readings` publishes it for a run's configuration.

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
from types import SimpleNamespace

import pytest
import torch
from test_artifact_invalidation import STANZA
from transformers import PretrainedConfig

import atom.compass.memory as memory_package
from atom.compass.artifacts import (
    ArtifactStore,
    GateState,
    Key,
    Kind,
    machine_spec_digest,
    machine_spec_key,
    member_name,
)
from atom.compass.backends.geometry import dtype_bytes as element_bytes
from atom.compass.backends.provenance import Method, combine
from atom.compass.memory import (
    MemoryReadings,
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
from atom.compass.run import memory as entry
from atom.compass.spec import MachineSpec

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
#: The model's stated parameter count, which sizes the meta builds below.
PARAMETERS = 27_000_000_000


def meta_tree(nbytes):
    """A meta build whose parameter storages hold `nbytes` and whose buffers are
    the 27B's rotary table, none of it resident."""
    tree = torch.nn.Module()
    tree.weight = torch.nn.Parameter(
        torch.empty(nbytes, dtype=torch.uint8, device="meta"), requires_grad=False
    )
    tree.register_buffer(
        "cos_sin", torch.empty(262_144, 64, dtype=torch.bfloat16, device="meta")
    )
    return tree


#: An 8B model at two bytes a parameter, the 8B the vertical slice's run file
#: states.
META_8B = meta_tree(16_000_000_000)

STACK = {"rocm": "7.2.4", "aiter": "f4e7c7509", "rccl": "2.22.3", "torch": "2.10.0"}

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
        "origin": "assumed",
    },
    "host": {
        "cpu": {"cores_physical": 96, "cores_logical": 192},
        "tokenizers": [TOKENIZER],
        "ipc": {"zmq_roundtrip_s": 5.0e-5, "shm_broadcast_s": 2.0e-5},
        "http_intake": {"fixed_s": 2.0e-3, "per_byte_s": 2.0e-8},
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
        "compute": {
            "bf16_flops": 2.5e15,
            "fp8_flops": 5.0e15,
            "derate": {"bf16": 0.70, "fp8": 0.70},
        },
        "software_pinned_to": dict(STACK),
    },
    "interconnect": {
        "intra_node": {
            "topology": "fully_connected",
            "domain_devices": 8,
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


#: One rank's member of the `memory_readings` entry at each width, as
#: `compass measure memory` publishes it: the reserve, residue and persistent
#: buffers at DOCUMENT's sizes, and the activations at its warmup tokens.
READINGS = {
    width: {
        "capacity_bytes": 288_000_000_000,
        "driver_and_collective_reserve_bytes": reserve,
        "allocator_retained_after_load_bytes": residue,
        "persistent_forward_buffer_bytes": 124_000_000,
        "activation_bytes": 805_306_368,
        "warmup_tokens": WARMUP_TOKENS,
        "cudagraph_pool_bytes": None,
        "record": "sha256:" + "e" * 64,
    }
    for width, reserve, residue in (
        (1, 970_000_000, 1_100_000),
        (2, 7_200_000_000, 2_170_000_000),
    )
}
SOURCE = "memory_readings test entry"


def measured(tp_width, **readings):
    """One rank's readings at `tp_width`, with `readings` moved."""
    return MemoryReadings.from_member({**READINGS[tp_width], **readings}, SOURCE)


def publish_readings(store, config, document=DOCUMENT, readings=READINGS[1]):
    """`config`'s entry in the store at `store`, one member a rank, keyed and
    conditioned as `compass measure memory` publishes it on the platform
    `document` states; the one already there, if there is one."""
    # As a run file carries it: JSON.
    machine = MachineSpec.from_mapping(json.loads(json.dumps(document)))
    fields = entry.key_fields(config)
    key = Key.of(
        Kind.MEMORY_READINGS, machine_spec=machine_spec_digest(machine), **fields
    )
    store = ArtifactStore(store)
    if not store.directory_for(key).exists():
        platform = machine_spec_key(machine).values
        platform["torch"] = machine.value("device.software_pinned_to.torch")
        store.publish(
            key,
            provenance=STANZA,
            topology=entry.topology(config),
            conditions=entry.conditions(fields, **platform),
            gates=GateState.of(),
            members={
                member_name(entry.STEM, rank, entry.EXTENSION): json.dumps(
                    readings
                ).encode()
                for rank in entry.topology(config).ranks()
            },
        )
    return key


#: What a worker's resolved config holds for the fields its memory key reads,
#: where a test's worker states none: a one-rank, unquantized deployment.
WORKER = {
    "pipeline_parallel_size": 1,
    "prefill_context_parallel_size": 1,
    "decode_context_parallel_size": 1,
    "enable_expert_parallel": False,
    "enable_dp_attention": False,
    "enable_tbo": False,
    "enable_tbo_decode": False,
    "max_num_seqs": 256,
    "max_model_len": 2048,
    "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
    "capture_sizes": list(CAPTURE_SIZES),
    "kv_cache_block_size": 16,
    "kv_cache_dtype": "bf16",
    # ATOM's `QuantizationConfig` resolves aiter dtypes, which needs a driver.
    "quant_config": SimpleNamespace(compute_hash=lambda: "unquantized-bf16"),
}


def with_readings(runner, store, readings=READINGS[1], document=DOCUMENT):
    """`runner`, a test's worker, given what its memory key reads and it does
    not state, with its configuration's entry, holding `readings` unless they
    are None, published in `store`."""
    config = runner.config
    for name, value in WORKER.items():
        if not hasattr(config, name):
            setattr(config, name, value)
    if not hasattr(config, "tp_world_size"):
        config.tp_world_size = config.tensor_parallel_size
    if not hasattr(config.parallel_config, "pipeline_parallel_rank"):
        config.parallel_config.pipeline_parallel_rank = 0
    if not hasattr(runner, "rank"):
        runner.rank = 0
    if readings is not None:
        publish_readings(store, config, document, readings)
    return runner


@pytest.fixture(autouse=True)
def no_device_readings(monkeypatch):
    """Both device readings raise, for every test in this file."""

    def refuse(*args, **kwargs):
        raise AssertionError(
            "a device reading was taken; this package promises that none is"
        )

    monkeypatch.setattr(torch.cuda, "mem_get_info", refuse)
    monkeypatch.setattr(torch.cuda, "memory_stats", refuse)


def spec_of(origin):
    """The suite's document, stated at one origin."""
    document = copy.deepcopy(DOCUMENT)
    document["provenance"]["origin"] = origin
    return MachineSpec.from_mapping(document)


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
    "bfloat16": {"weights": 54_000_000_000},
    "float32": {"weights": 108_000_000_000},
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


def model_terms(tree=META_8B):
    """`tree`'s model terms, the activations off the TP1 entry."""
    return ModelTerms.from_tree(
        tree, activations=measured(1).activations(WARMUP_TOKENS)
    )


def readings_at(spec, qwen, tp_width):
    total_bytes = int(spec.value("device.memory.capacity_bytes"))
    return device_readings(
        spec,
        memory=measured(tp_width),
        tp_width=tp_width,
        model=model_terms(
            meta_tree(PARAMETERS * element_bytes(qwen.dtype) // tp_width)
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
    for term, reading in zip(
        readings.free.terms, (readings.total, readings.peak_torch, readings.non_torch)
    ):
        assert term.method is combine(t.method for t in reading.terms)


@pytest.mark.parametrize("tp_width", [1, 2])
def test_every_term_names_where_it_came_from(spec, qwen, tp_width):
    readings = readings_at(spec, qwen, tp_width)
    for name, reading in readings.as_dict().items():
        for term in reading.terms:
            assert term.source.strip(), f"{name}.{term.name}"
            if term.source.startswith("device."):
                path = term.source.partition("[")[0].partition(" x ")[0]
                assert path in spec.values, f"{name}.{term.name} names {path!r}"
            if term.method is Method.DECLARED:
                assert term.note.strip(), f"{name}.{term.name}"


def test_on_a_probed_spec_no_term_is_declared(qwen):
    # The weights come off a meta build and the activations off the entry.
    readings = readings_at(spec_of("probed"), qwen, 1)
    assert readings.declared == ()
    assert readings.peak_torch.terms[-1].method is Method.MEASURED


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


def test_a_configuration_that_does_not_fit_refuses_rather_than_clamping(spec, qwen):
    with pytest.raises(MemoryRefusal) as refusal:
        device_readings(
            spec,
            memory=measured(1),
            tp_width=1,
            model=model_terms(meta_tree(800_000_000_000)),
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
            memory=measured(1),
            tp_width=1,
            model=model_terms(),
            cudagraph_overhead=predicts(measured(1, cudagraph_pool_bytes=420_000_000)),
        )
    assert "4-19x" in str(refusal.value)


def test_the_predicted_pool_is_the_one_the_entrys_start_reserved():
    # What the pool really costs is read, not projected: the reserved bytes
    # the entry's start added across its own capture, at its own ladder.
    (term,) = predicts(measured(2, cudagraph_pool_bytes=420_000_000)).terms
    assert (term.nbytes, term.method) == (420_000_000, Method.MEASURED)
    assert term.source == f"{SOURCE} cudagraph_pool_bytes"
    assert term.records == (READINGS[2]["record"],)
    # A start that captured nothing has no pool to predict.
    assert predicts(measured(2)) is None


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
    reading = Reading("x", (Term("a", 3, Method.ANALYTICAL, "somewhere"),))
    assert not hasattr(reading, "__int__")
    assert not hasattr(reading, "__index__")
    assert "a" in str(reading) and "somewhere" in str(reading)


def test_a_term_without_a_source_is_refused():
    with pytest.raises(ValueError, match="states no source"):
        Term("a", 3, Method.MEASURED, "  ")


def test_a_declared_term_must_say_what_replaces_it():
    with pytest.raises(ValueError, match="a: a declared answer needs a note"):
        Term("a", 3, Method.DECLARED, "a coefficient")


def test_a_reading_with_no_terms_is_refused():
    with pytest.raises(ValueError, match="has no terms"):
        Reading("x", ())


def test_the_total_follows_the_terms_rather_than_being_stored():
    terms = (
        Term("a", 10, Method.MEASURED, "device.memory.capacity_bytes"),
        Term("b", -4, Method.ESTIMATED, "a reading"),
    )
    assert Reading("x", terms).total == 6


# --- the assumptions this module makes, shown rather than held ---------------


def test_the_model_terms_follow_the_element_size(spec, resident):
    name, config = resident
    peak = readings_at(spec, config, 1).peak_torch
    expected = MODEL_TERMS_BY_DTYPE[name]
    assert {t.name: t.nbytes for t in peak.terms if t.name in expected} == expected


def test_the_graph_pool_reservation_follows_the_element_size(resident):
    name, config = resident
    assert reserved(config, 288.0e9).total == GRAPH_POOL_BY_DTYPE[name]


def test_the_entry_measures_the_activations_at_its_own_warmup_and_scales_them():
    # One start reads `peak - current` over its warmup: measured at those
    # tokens, and the coefficient times another count is an estimate.
    term = measured(1).activations(WARMUP_TOKENS)
    assert (term.nbytes, term.method) == (805_306_368, Method.MEASURED)
    assert (
        term.source == f"{SOURCE} activation_bytes over {WARMUP_TOKENS} warmup tokens"
    )
    half = measured(1).activations(WARMUP_TOKENS // 2)
    assert (half.nbytes, half.method) == (402_653_184, Method.ESTIMATED)
    assert not half.validated


@pytest.mark.parametrize(
    "readings",
    [
        {"warmup_tokens": 0},
        {"activation_bytes": 1.5},
        {"capacity_bytes": None},
        {"persistent_forward_buffer_bytes": True},
        {"made_up_bytes": 1},
        {"record": 5},
        {"record": "a log"},
    ],
)
def test_a_member_that_is_not_one_ranks_readings_is_refused(readings):
    with pytest.raises(MemoryRefusal, match=SOURCE):
        measured(1, **readings)


#: What a member published outside `compass measure memory` states for a
#: platform no engine has started on.
ASSUMED = "copied from MI355X's memory_readings entry of this configuration"


def test_an_assumed_member_declares_every_term_built_on_it(qwen):
    # On a probed spec, so what is declared is the entry's and nothing else.
    readings = device_readings(
        spec_of("probed"),
        memory=measured(1, assumed=ASSUMED),
        tp_width=1,
        model=ModelTerms.from_tree(
            META_8B,
            activations=measured(1, assumed=ASSUMED).activations(WARMUP_TOKENS),
        ),
        cudagraph_overhead=reserved(qwen, 288.0e9),
    )
    assert readings.declared == (
        "peak_torch.load residue",
        "peak_torch.persistent",
        "peak_torch.activations",
        "non_torch.driver and collective reserve",
        "free.less peak_torch",
        "free.less non_torch",
    )
    for term in (*readings.peak_torch.terms[2:], *readings.non_torch.terms):
        assert term.note.startswith(f"{ASSUMED}; a compass measure memory start")
    pool = predicts(measured(1, assumed=ASSUMED, cudagraph_pool_bytes=420_000_000))
    assert pool.declared == ("pool",)


@pytest.mark.parametrize("assumed", ["", "  ", None, 7])
def test_an_assumption_that_names_nothing_is_refused(assumed):
    with pytest.raises(MemoryRefusal, match="which names no assumption"):
        measured(1, assumed=assumed)


def test_the_negative_box_refusal_carries_its_decomposition(spec, qwen):
    # A refusal carries its decomposition too: the reader has to see which of the
    # six terms does not fit, and three of them are declared coefficients.
    with pytest.raises(MemoryRefusal) as refusal:
        device_readings(
            spec,
            memory=measured(1),
            tp_width=1,
            model=model_terms(meta_tree(800_000_000_000)),
            cudagraph_overhead=reserved(qwen, 288.0e9),
        )
    message = str(refusal.value)
    for term in ("weights", "buffers", "load residue", "persistent", "activations"):
        assert term in message, term
    assert "gpu-memory-utilization" in message


def test_the_eager_branch_mirrors_atom_and_is_analytical():
    # ATOM reserves nothing under enforce_eager; mirroring that is exact.
    assert reserves(enforce_eager=True).terms[0].method is Method.ANALYTICAL


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
