# SPDX-License-Identifier: MIT
"""The pipeline-stage block count, re-derived rather than restated.

The memory and KV design document used to say that the `all_reduce(MIN)`
across pipeline stages in `get_num_blocks` is inert, *because every stage
computes the same number*, and that it *still needs a live process group or a
stub*. Both halves were wrong, and prose is how they stayed wrong: nothing
re-checked them.

So this file checks them. The split is never written out here: it comes from
ATOM's own `get_pp_indices`, which hands the remainder to the *middle*
partitions and so produces a layout no reader would guess.
`VLLM_PP_LAYER_PARTITION` overrides
that partitioner, so every derivation clears it first; left set, this reads a
layer layout out of the environment and calls it ATOM's.

The budget, the block size and the widths are choices, not measurements, and
they are named constants for that reason: what is derived here is how the
count *moves* between stages, not its absolute value. The model is the
64-layer hybrid vendored beside this file -- one full-attention layer in four,
so 16 of the 64 hold paged KV. A stage's paged-layer count is therefore not a
function of the layers it holds, which is why both are derived and both are
pinned.

The guard on the reduce is read out of ATOM's source with `ast` rather than by
importing it: the claim is about what the code says, and the runner does not
import on a machine with no driver. It is matched by attribute name and scoped
to `get_num_blocks`, not matched against the text `torch.distributed.all_reduce`
-- the runner reaches the same function through an aliased `dist.all_reduce`
elsewhere, so a spelling-bound matcher would be making a claim about a spelling
while sounding like one about the file.
"""

import ast
import json
import pathlib

from transformers import PretrainedConfig

from atom.compass.backends import KvGeometry, Parallelism
from atom.model_ops.attentions.pool_layout.sub_pool_spec import page_pool, plan_pools
from atom.models.utils import get_pp_indices

REPO = pathlib.Path(__file__).resolve().parents[2]
RUNNER = REPO / "atom/model_engine/model_runner.py"
CONFIG_JSON = pathlib.Path(__file__).with_name("qwen3_5_27b_config.json")
QWEN = PretrainedConfig.from_dict(json.loads(CONFIG_JSON.read_text())["text_config"])
BLOCK_SIZE = 64
MAX_NUM_SEQS = 256
KV_BUDGET_BYTES = 200_000_000_000
WIDTHS = range(2, 9)


def stages(pp_size, monkeypatch):
    """(layers held, paged layers, blocks) per stage, as the runner splits it."""
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    spans = [
        get_pp_indices(int(QWEN.num_hidden_layers), rank, pp_size)
        for rank in range(pp_size)
    ]
    geometries = [
        KvGeometry.from_hf_config(
            QWEN,
            block_size=BLOCK_SIZE,
            parallelism=Parallelism(pp_size=pp_size),
            layer_range=span,
        )
        for span in spans
    ]
    return [
        (
            end - start,
            geometry.layers,
            plan_pools(
                [page_pool(geometry.bytes_per_block)], KV_BUDGET_BYTES, MAX_NUM_SEQS
            ).paged_entries,
        )
        for (start, end), geometry in zip(spans, geometries)
    ]


def test_the_stages_do_not_all_compute_the_same_block_count(monkeypatch):
    """The stages, taken from ATOM's own partitioner.

    Three stages of a 64-layer stack hold 21, 22 and 21 layers, which is 5, 5
    and 6 paged ones -- **two** distinct block counts, not three: the two
    five-layer stages share one. An early paraphrase of this said three, so
    the counts are asserted as a list rather than as a cardinality.

    Every width from 2 to 8 is taken, because the reason given for the
    reduction being inert -- that the readings are identical across stages --
    would hold at all of them. It is inert at 2, 4 and 8, and nowhere else
    here.

    The layers held are carried beside the paged count because the two are
    not the same function: at six stages, stages 1 and 3 each hold 11 layers
    and hold 3 and 2 paged ones, and only the second number moves the count.
    """
    per_width = {pp: stages(pp, monkeypatch) for pp in WIDTHS}
    assert per_width[3] == [(21, 5, 152587), (22, 5, 152587), (21, 6, 127156)]
    assert per_width[6] == [
        (10, 2, 381469),
        (11, 3, 254313),
        (11, 3, 254313),
        (11, 2, 381469),
        (11, 3, 254313),
        (10, 3, 254313),
    ]
    assert {
        pp: len({blocks for _, _, blocks in rows}) for pp, rows in per_width.items()
    } == {2: 1, 3: 2, 4: 1, 5: 2, 6: 2, 7: 2, 8: 1}


def test_the_reduce_is_guarded_so_it_needs_neither_a_group_nor_a_stub():
    """The reduce runs only when a process group is already initialised.

    The assertion is not merely that a guard exists somewhere near the
    reduce, which a later edit could leave true while moving the call out from
    under it. It is that the only collective `get_num_blocks` performs sits
    inside the `is_initialized()` branch -- so with no group nothing runs, and
    nothing has to be stubbed for it.

    Both comprehensions are scoped to `get_num_blocks` and match the called
    attribute rather than a dotted path. The runner calls the same function as
    `dist.all_reduce` through a local alias in an unrelated method, and that
    call is guarded by a data-parallel width instead; a matcher keyed on the
    text `torch.distributed.all_reduce` would miss it, and would also miss a
    second one added here under the alias.
    """
    sized = [
        node
        for node in ast.walk(ast.parse(RUNNER.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "get_num_blocks"
    ]
    guards = [
        node
        for func in sized
        for node in ast.walk(func)
        if isinstance(node, ast.If)
        and "is_initialized()" in ast.unparse(node.test)
        and "ReduceOp.MIN" in ast.unparse(node)
    ]
    reduces = [
        node
        for func in sized
        for node in ast.walk(func)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "all_reduce"
    ]
    assert sized and len(guards) == 1 and len(reduces) == 1
    assert {id(node) for node in ast.walk(guards[0])} >= {id(n) for n in reduces}
