# SPDX-License-Identifier: MIT
"""A block count out of ATOM's own budget arithmetic, with the card unread.

This is the one file in `tests/compass/` that needs a driver, and it needs one
for a reason that has nothing to do with reading a device: importing
`atom.model_engine.model_runner` runs aiter's architecture probe, which shells
out to `rocminfo`. So the gate it runs in is decided by an import, not by the test --
which is why every device reading is patched to raise for every test here. The
pool is sized on a machine that has a card, from a spec describing a different
one, and the card that is present is never asked anything.

The runner is built with `object.__new__` and given the attributes
`get_num_blocks` reads. Constructing one for real would load a checkpoint,
build a module tree and open a process group, none of which this is about. The
budget arithmetic that runs is ATOM's own, unmodified and unwrapped: this file
calls `CompassModelRunner.get_num_blocks`, whose body is a refusal check, a
`super()` call and a record.

The entry size the pool is planned from comes from `backends/geometry`, not
from an attention builder, because a builder is attached to a constructed
engine. That is the declaration `plan_pools` consumes; `plan_pools` itself, and
every line of budget arithmetic above it, is ATOM's.

Both halves of the sizing are built at the same width, and the entry size is
pinned as well as the count so that they cannot drift apart again: the readings
describe one rank's memory, so the block they are divided by has to be one
rank's block. Nothing downstream of `plan_pools` can tell that it was not --
the count comes back plausible either way.
"""

import copy
import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch
from test_memory_readings import (
    CONFIG_JSON,
    DOCUMENT,
    GPU_MEMORY_UTILIZATION,
    readings_at,
)
from transformers import PretrainedConfig

from atom.compass.backends.geometry import KvGeometry, Parallelism
from atom.compass.backends.provenance import Method
from atom.compass.memory import Reading, Term
from atom.compass.spec import MachineSpec
from atom.model_ops.attentions.pool_layout.sub_pool_spec import page_pool

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="importing the engine runs aiter's architecture probe, which needs a "
    "driver; the sizing itself reads no device and every device read is patched "
    "to raise",
)

BLOCK_SIZE = 64
MAX_NUM_SEQS = 256
MAX_MODEL_LEN = 32768

#: The block counts this file produced on node 18, container
#: `xiaobizh_n18`, 2026-09-22, pinned so that a term moving anywhere
#: upstream of it is a failure here rather than a different number nobody
#: compared. The arithmetic behind them is ATOM's: this file adds none.
EXPECTED_BLOCKS = {1: 46641, 2: 102152}

#: What one block costs the rank being modelled, pinned beside the count.
#: A TP2 row was once priced against a TP1 block: the readings were
#: per-rank and the geometry was not, so a footprint for one rank was divided
#: by a block belonging to the whole model. Both halves of the reply now state
#: the width they were built at, and the entry size is the half that would
#: have shown it -- 4 KV heads over 2 ranks halves the block exactly, so the
#: count was out by a clean factor the reader had nothing to check it against.
EXPECTED_ENTRY_BYTES = {1: 4_194_304, 2: 2_097_152}


@pytest.fixture(autouse=True)
def no_device_readings(monkeypatch):
    """Every device reading raises, for every test in this file."""

    def refuse(*args, **kwargs):
        raise AssertionError(
            "a device reading was taken; this file sizes the budget from a "
            "spec and must never read the card that is present"
        )

    monkeypatch.setattr(torch.cuda, "mem_get_info", refuse)
    monkeypatch.setattr(torch.cuda, "memory_stats", refuse)
    monkeypatch.setattr(torch.cuda, "memory_reserved", refuse)


@pytest.fixture(scope="module")
def spec():
    return MachineSpec.from_mapping(copy.deepcopy(DOCUMENT))


@pytest.fixture(scope="module")
def qwen():
    return PretrainedConfig.from_dict(
        json.loads(CONFIG_JSON.read_text())["text_config"]
    )


class _Builder:
    """The one declaration `get_num_blocks` asks a builder for, plus its state.

    A real builder is attached to a constructed engine and reads its weights'
    dtype off a loaded model. This declares the same thing -- one PAGE class at
    the entry size this model's KV costs -- and no state transfer, which is the
    case every deployment but PAGE-backed checkpointing is in.
    """

    def __init__(self, entry_bytes):
        self._entry_bytes = entry_bytes

    def sub_pool_specs(self):
        return [page_pool(self._entry_bytes)]

    def state_transfer(self):
        from atom.model_engine.state_runtime import StateTransfer

        return StateTransfer.none()


def _runner(spec, qwen, tp_width, *, utilisation=GPU_MEMORY_UTILIZATION, free=None):
    """A `CompassModelRunner` with the attributes the budget method reads."""
    from atom.compass.runner.model_runner import CompassModelRunner
    from atom.compass.runner.overrides import install_device_readings

    readings = readings_at(spec, qwen, tp_width)
    if free is not None:
        readings = dataclasses.replace(
            readings,
            free=Reading(
                "free",
                (
                    Term(
                        "probe",
                        int(free),
                        Method.ANALYTICAL,
                        "a probe that moves the clamp, not a reading",
                    ),
                ),
            ),
        )
    geometry = KvGeometry.from_hf_config(
        qwen,
        block_size=BLOCK_SIZE,
        parallelism=Parallelism(tp_size=tp_width),
    )
    runner = object.__new__(CompassModelRunner)
    runner.device = torch.device("cuda", 0)
    runner.block_size = BLOCK_SIZE
    runner.attn_metadata_builder = _Builder(geometry.bytes_per_block)
    runner.config = SimpleNamespace(
        hf_config=qwen,
        gpu_memory_utilization=utilisation,
        max_num_seqs=MAX_NUM_SEQS,
        max_model_len=MAX_MODEL_LEN,
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        enforce_eager=False,
        enable_rapidserve=False,
        disagg_is_decode=False,
    )
    install_device_readings(runner, readings)
    return runner


# --- the block count ---------------------------------------------------------


@pytest.mark.parametrize("tp_width", [1, 2])
def test_the_pool_is_sized_by_atoms_arithmetic_with_the_card_unread(
    spec, qwen, tp_width, capsys
):
    """The block count, its four keys, and the record it carries.

    `state_runtime` is put back through `StateRuntime.from_wire`, which is what
    `EngineCore.__init__` does with it and which raises unless its key set is
    exactly the two it wants -- so a reply that would fail on the first RPC of
    the engine's life fails here instead.
    """
    from atom.model_engine.state_runtime import StateRuntime

    runner = _runner(spec, qwen, tp_width)
    reply = runner.get_num_blocks()

    assert set(reply) == {
        "num_kvcache_blocks",
        "pool_entries",
        "pool_entries_per_req",
        "state_runtime",
    }
    assert reply["num_kvcache_blocks"] == EXPECTED_BLOCKS[tp_width]
    assert reply["num_kvcache_blocks"] > 0
    assert StateRuntime.from_wire(reply["state_runtime"]).transfer.copies is False
    assert reply["pool_entries"] == {"kv": EXPECTED_BLOCKS[tp_width]}
    # The block the count is a count *of*, pinned beside it: the readings are
    # per-rank, so the geometry has to be too, and nothing downstream of here
    # can tell that it was not.
    assert runner.pool_plan.entry_bytes["kv"] == EXPECTED_ENTRY_BYTES[tp_width]

    sizing = runner.kv_pool_sizing
    assert sizing.num_kvcache_blocks == reply["num_kvcache_blocks"]
    # Every subtracted term is read off the meta build or the entry.
    assert sizing.declared_terms == ()
    with capsys.disabled():
        print()
        print(sizing.table())


# --- the clamp, proved inert rather than asserted inert ----------------------


@pytest.mark.parametrize("utilisation", [0.5, 0.9])
def test_the_min_budget_free_clamp_does_not_bind_on_either_side_of_free(
    spec, qwen, utilisation
):
    """`free` is a clean box, so the clamp cannot be what decides the count.

    Proved by removing it rather than by re-deriving the budget here: sizing
    the same pool again with `free` set far above anything the budget could
    reach gives the same count, so the real `free` was not the binding limit in
    the run beside it. No arithmetic in this test is ATOM's, because there is
    none.

    Run at two utilisations, chosen so that the utilisation budget itself is
    below `free` at one and above it at the other -- the regime the clamp was
    written for is the second, and it does not bind there either.
    """
    readings = readings_at(spec, qwen, 1)
    budget = int(readings.total.total * utilisation)
    below = budget < readings.free.total
    assert below == (utilisation == 0.5), "the two runs must straddle free"

    real = _runner(spec, qwen, 1, utilisation=utilisation).get_num_blocks()
    unclamped = _runner(
        spec, qwen, 1, utilisation=utilisation, free=readings.total.total * 1000
    ).get_num_blocks()
    assert real["num_kvcache_blocks"] == unclamped["num_kvcache_blocks"]


def test_the_probe_sees_the_clamp_when_it_does_bind(spec, qwen):
    """Otherwise a probe that changed nothing would read as an inert clamp for
    any budget at all."""
    real = _runner(spec, qwen, 1).get_num_blocks()
    clamped = _runner(spec, qwen, 1, free=10_000_000_000).get_num_blocks()
    assert clamped["num_kvcache_blocks"] < real["num_kvcache_blocks"]


# --- what the runner refuses rather than sizes, on the class itself ----------


def test_the_composed_class_refuses_the_disagg_decode_process(spec, qwen):
    """The refusal is reachable through the real class and its real MRO, not
    only through the mixin's function."""
    from atom.compass.runner.overrides import RunnerRefusal

    runner = _runner(spec, qwen, 1)
    runner.config.disagg_is_decode = True
    with pytest.raises(RunnerRefusal, match="owns no device memory"):
        runner.get_num_blocks()


def test_rapidserve_prefill_holds_back_what_atoms_rapidserve_runner_does(spec, qwen):
    """Prefill's pool is ATOM's RapidServe runner's on the same readings, and
    smaller than a dedicated card's."""
    from atom.compass.runner.model_runner import CompassRapidServeModelRunner
    from atom.model_engine.model_runner import RapidServeModelRunner

    plain = _runner(spec, qwen, 1)
    rapid = object.__new__(CompassRapidServeModelRunner)
    real = object.__new__(RapidServeModelRunner)
    rapid.__dict__.update(vars(plain))
    real.__dict__.update(vars(plain))
    real._read_device_memory = rapid._read_device_memory
    real._estimate_cudagraph_overhead = rapid._estimate_cudagraph_overhead
    plain_n, rapid_n, real_n = (
        r.get_num_blocks()["num_kvcache_blocks"] for r in (plain, rapid, real)
    )
    assert rapid_n == real_n < plain_n
