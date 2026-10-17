# SPDX-License-Identifier: MIT
"""The stand-in model's KV geometry, checked by what ATOM does with it.

The geometry itself is a handful of multiplications, and a test that only
re-derives them proves nothing: it would pass against a number the engine
refuses. So every claim below is made by handing the geometry to the code
that consumes it for real -- `page_pool` and `plan_pools` from
`atom.model_ops.attentions`, `BlockManager` from `atom.model_engine` and ATOM's
own pipeline partitioner -- and reading the answer back out of those.

The model is the published Qwen3.8-27B config, vendored beside this file so
the numbers are reproducible without a checkout of the weights. It is a
hybrid: 64 layers of which 16 are full attention, so only those 16 hold paged
KV, and it has 4 KV heads, which is what makes the grouped-query bound visible
at the widths a real deployment uses. At 8 ranks those 4 heads cannot be cut
into 8, so they are replicated and a rank's KV stops shrinking. A geometry
that divided by the width would claim twice the blocks there.

What is deliberately not asserted: agreement with a real run's block count.
The double leaves out the fp32 scale plane the aiter backends carry and sizes
no recurrent state, so its numbers are smaller than a real model's by a
declared amount. What it has to get right is that the count moves, correctly,
with the width.

One asymmetry to carry forward. At one pipeline stage the 16 can be checked
against the engine twice over -- it is the count of full-attention entries in
the config, and it is also `num_hidden_layers // full_attention_interval`,
which is what the GDN builder sizes its paged pool from. At more than one
stage there is no such check available, because ATOM's own GDN path mixes a
pipeline-local layer count with a global one there and sizes a negative
block. The four-per-stage figure below is the intended semantics and is
right; it is not agreement with a live engine, and nothing here claims it is.
"""

import json
import pathlib

import pytest
from conftest import atom_config_double
from transformers import PretrainedConfig

from atom.compass.backends import KvGeometry, Parallelism
from atom.model_engine.block_manager import BlockManager
from atom.model_ops.attentions.pool_layout.sub_pool_spec import page_pool, plan_pools
from atom.models.utils import get_pp_indices

CONFIG_JSON = pathlib.Path(__file__).with_name("qwen3_5_27b_config.json")
BLOCK_SIZE = 64
MAX_NUM_SEQS = 512
# A budget, not a measurement: no device is read for it, and a fixed number is
# what makes these counts reproducible.
KV_BUDGET_BYTES = 64 << 30


@pytest.fixture(scope="module")
def qwen():
    raw = json.loads(CONFIG_JSON.read_text())
    return PretrainedConfig.from_dict(raw["text_config"])


def blocks_for(geometry, budget=KV_BUDGET_BYTES):
    """The block count ATOM's own pool sizing gives this geometry."""
    plan = plan_pools([page_pool(geometry.bytes_per_block)], budget, MAX_NUM_SEQS)
    return plan.paged_entries


def block_manager_for(geometry, budget=KV_BUDGET_BYTES):
    """A real BlockManager sized from this geometry."""
    plan = plan_pools([page_pool(geometry.bytes_per_block)], budget, MAX_NUM_SEQS)
    config = atom_config_double(
        num_kvcache_blocks=plan.paged_entries,
        kv_cache_block_size=BLOCK_SIZE,
        max_num_seqs=MAX_NUM_SEQS,
    )
    config.pool_entries = dict(plan.entries)
    config.pool_entries_per_req = dict(plan.entries_per_req)
    return BlockManager(config)


def test_only_the_attention_layers_of_the_stage_hold_paged_kv(qwen):
    geometry = KvGeometry.from_hf_config(qwen, block_size=BLOCK_SIZE)
    assert qwen.num_hidden_layers == 64
    assert geometry.layers == 16
    assert geometry.kv_heads == 4
    assert geometry.bytes_per_block == 16 * BLOCK_SIZE * 2 * 4 * 256 * 2


@pytest.mark.parametrize("tp_size, kv_heads", [(1, 4), (2, 2), (4, 1), (8, 1)])
def test_tensor_parallel_shards_kv_heads_until_the_gqa_bound(qwen, tp_size, kv_heads):
    geometry = KvGeometry.from_hf_config(
        qwen, block_size=BLOCK_SIZE, parallelism=Parallelism(tp_size=tp_size)
    )
    assert geometry.kv_heads == kv_heads
    # The bound is the config's KV head count, not the query heads: dividing
    # by those would give 24/tp and a block 6x too large.
    assert geometry.kv_heads != qwen.num_attention_heads // tp_size


def test_atom_sizes_a_different_pool_at_every_tp_width(qwen):
    counts = {
        tp: blocks_for(
            KvGeometry.from_hf_config(
                qwen, block_size=BLOCK_SIZE, parallelism=Parallelism(tp_size=tp)
            )
        )
        for tp in (1, 2, 4, 8)
    }
    assert counts[2] == 2 * counts[1]
    assert counts[4] == 4 * counts[1]
    # The grouped-query bound, seen from the far end: a rank at 8 holds the
    # same replicated head it held at 4, so the pool does not double again.
    assert counts[8] == counts[4]


def test_the_block_manager_runs_on_the_count_the_geometry_produced(qwen, seq_factory):
    geometry = KvGeometry.from_hf_config(
        qwen, block_size=BLOCK_SIZE, parallelism=Parallelism(tp_size=8)
    )
    manager = block_manager_for(geometry)
    assert manager.kv.num_free == blocks_for(geometry)
    sequence = seq_factory(list(range(3 * BLOCK_SIZE)), block_size=BLOCK_SIZE)
    assert manager.can_allocate(sequence) >= 0
    manager.allocate(sequence)
    assert len(sequence.block_table) == 3
    assert manager.kv.num_free == blocks_for(geometry) - 3


def test_a_pool_too_small_for_one_block_is_refused_by_the_block_manager(qwen):
    geometry = KvGeometry.from_hf_config(qwen, block_size=BLOCK_SIZE)
    assert blocks_for(geometry, budget=geometry.bytes_per_block - 1) == 0
    with pytest.raises(AssertionError):
        block_manager_for(geometry, budget=geometry.bytes_per_block - 1)


def test_each_pipeline_stage_is_sized_from_the_layers_atom_gives_it(qwen):
    whole = blocks_for(KvGeometry.from_hf_config(qwen, block_size=BLOCK_SIZE))
    parallelism = Parallelism(pp_size=4)
    staged = [
        KvGeometry.from_hf_config(
            qwen,
            block_size=BLOCK_SIZE,
            parallelism=parallelism,
            layer_range=get_pp_indices(qwen.num_hidden_layers, rank, 4),
        )
        for rank in range(4)
    ]
    assert [g.layers for g in staged] == [4, 4, 4, 4]
    assert [blocks_for(g) for g in staged] == [4 * whole] * 4
    assert block_manager_for(staged[0]).kv.num_free == 4 * whole


def test_a_declared_pipeline_stage_must_say_which_one_it_is(qwen):
    with pytest.raises(ValueError, match="no layer_range"):
        KvGeometry.from_hf_config(
            qwen, block_size=BLOCK_SIZE, parallelism=Parallelism(pp_size=4)
        )


def test_data_and_expert_parallel_leave_the_kv_block_alone(qwen):
    alone = KvGeometry.from_hf_config(
        qwen, block_size=BLOCK_SIZE, parallelism=Parallelism(tp_size=2)
    )
    wide = Parallelism(tp_size=2, dp_size=4, expert_parallel=True)
    assert (
        KvGeometry.from_hf_config(qwen, block_size=BLOCK_SIZE, parallelism=wide)
        == alone
    )


def test_expert_parallelism_alone_builds_no_all_to_all():
    assert Parallelism(expert_parallel=True).collectives() == ()
    assert Parallelism(tp_size=8, expert_parallel=True).collectives() == (
        "tp-all-reduce",
    )
    assert Parallelism(tp_size=8, dp_size=2, expert_parallel=True).collectives() == (
        "tp-all-reduce",
        "moe-all-to-all",
    )
    assert Parallelism(dp_size=2).collectives() == ()


def test_a_width_the_kv_heads_cannot_shard_is_refused(qwen):
    """Which of the two indivisible cases is refused is the whole content.

    `kv_heads_per_rank` refuses on either side of `total_kv_heads >= tp_size`,
    and the two sentences differ only in which width is said not to divide the
    other. Reading that comparison the other way up routes this very call to
    the replicated branch, so `do not divide` alone held both ways. The whole
    sentence is matched, from its first character to its last, so the operand
    order is asserted and the replicated branch's extra clause cannot be
    present. The next test reaches that branch and holds its sentence.
    """
    heads = qwen.num_key_value_heads
    with pytest.raises(
        ValueError, match=rf"^{heads} KV heads do not divide across 3 ranks$"
    ):
        KvGeometry.from_hf_config(
            qwen, block_size=BLOCK_SIZE, parallelism=Parallelism(tp_size=3)
        )


def test_more_ranks_than_kv_heads_that_they_do_not_divide_is_refused(qwen):
    """Six ranks over 4 KV heads: the heads cannot be cut and cannot be copied.

    More ranks than heads is the replicated case, where each rank keeps a copy
    of one head. That needs the heads to divide the ranks, and 4 does not
    divide 6, so this is refused too, with the operands the other way round.
    """
    heads = qwen.num_key_value_heads
    with pytest.raises(
        ValueError,
        match=rf"^6 ranks do not divide across {heads} KV heads, so the "
        r"replicated case does not apply either$",
    ):
        KvGeometry.from_hf_config(
            qwen, block_size=BLOCK_SIZE, parallelism=Parallelism(tp_size=6)
        )


def test_an_unrecognised_layer_kind_is_refused_rather_than_paged(qwen):
    """A windowed kind keeps a bounded cache, which this geometry cannot size."""
    windowed = PretrainedConfig.from_dict(
        {**qwen.to_dict(), "layer_types": ["sliding_attention"] * 64}
    )
    with pytest.raises(ValueError, match="sliding_attention"):
        KvGeometry.from_hf_config(windowed, block_size=BLOCK_SIZE)


def test_a_mixed_stack_that_names_no_layer_kinds_is_refused(qwen):
    """The failure this catches sizes four times the bytes and still starts."""
    unstated = qwen.to_dict()
    del unstated["layer_types"]
    unstated["full_attention_interval"] = 4
    with pytest.raises(ValueError, match="full_attention_interval"):
        KvGeometry.from_hf_config(
            PretrainedConfig.from_dict(unstated), block_size=BLOCK_SIZE
        )


def test_a_uniform_stack_needs_no_layer_kinds():
    """A dense model names none, and every one of its layers holds a cache."""
    dense = PretrainedConfig.from_dict(
        {
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "num_attention_heads": 32,
            "hidden_size": 4096,
            "dtype": "float16",
        }
    )
    geometry = KvGeometry.from_hf_config(dense, block_size=BLOCK_SIZE)
    assert (geometry.layers, geometry.head_dim, geometry.element_bytes) == (32, 128, 2)
    assert blocks_for(geometry) == KV_BUDGET_BYTES // geometry.bytes_per_block


def test_a_config_with_no_dtype_is_refused_rather_than_sized(qwen):
    """With no dtype and no kv_dtype there is no element size to assume."""
    nameless = PretrainedConfig.from_dict({**qwen.to_dict(), "dtype": None})
    with pytest.raises(ValueError, match="states no `dtype` and no kv_dtype"):
        KvGeometry.from_hf_config(nameless, block_size=BLOCK_SIZE)
    stated = KvGeometry.from_hf_config(
        nameless, block_size=BLOCK_SIZE, kv_dtype="float16"
    )
    assert stated.element_bytes == 2
