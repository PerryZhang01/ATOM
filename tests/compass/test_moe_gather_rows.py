# SPDX-License-Identifier: MIT
"""With expert parallelism off, the MoE segment is priced from the rows the DP
gather carries, settled by each rank's own `ForwardMode.decide` over a real
eight-rank gloo group.

Every rank decoding, the padded `all_gather` carries `dp x running_bs x
max_seqlen_q`, `running_bs` the rung at or above the group's largest batch.
Any rank prefilling, the variable-length gather carries `sum(num_tokens_across_dp)`.
"""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import MockConfig
from test_dp_step_max import BLOCK, Recording, prefill, runner

import atom.utils.distributed.utils as dist_utils
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams
from atom.utils import get_open_port

DP = 8
LADDER = np.array([1, 2, 4, 8], dtype=np.int32)


def decode(width):
    """The second batch ATOM's scheduler builds for `width` four-token prompts."""
    scheduler = Scheduler(
        MockConfig(
            num_kvcache_blocks=64,
            kv_cache_block_size=BLOCK,
            max_model_len=256,
            max_num_seqs=8,
            parallel_config=SimpleNamespace(data_parallel_rank=0),
        )
    )
    for _ in range(width):
        scheduler.add(Sequence([5, 6, 7, 8], BLOCK, sampling_params=SamplingParams()))
    batch, seqs = scheduler.schedule()
    scheduler.postprocess(
        list(seqs.values()),
        ScheduledBatchOutput(
            req_ids=list(batch.req_ids),
            token_ids=[(7,) for _ in batch.req_ids],
            num_rejected=None,
            num_bonus=None,
            draft_token_ids=None,
        ),
        batch=batch,
    )
    batch = scheduler.schedule()[0]
    assert batch.total_seqs_num_decode == width
    return batch


@pytest.fixture
def group(monkeypatch):
    """Price one batch per rank of an eight-rank group; return (seconds, view) per rank."""
    monkeypatch.setattr(
        dist_utils, "_get_default_timeout", lambda _: timedelta(seconds=10)
    )
    # Gloo binds the address the hostname resolves to, one DNS lookup per rank;
    # a stalled lookup outlasts the timeout above. Loopback needs no lookup.
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    local = threading.local()
    monkeypatch.setitem(
        sys.modules,
        "aiter.dist.parallel_state",
        SimpleNamespace(get_dp_group=lambda: SimpleNamespace(cpu_group=local.group)),
    )

    def run(batches):
        port = get_open_port()

        def on_rank(rank):
            backend = Recording()
            r = runner(backend, dp_size=DP, dp_rank=rank)
            r.capture_sizes_np = LADDER
            local.group = dist_utils.stateless_init_torch_distributed_process_group(
                "127.0.0.1", port, rank, DP, backend="gloo"
            )
            try:
                return r.forward(batches[rank]).predicted_s, backend.views[0]
            finally:
                dist_utils.stateless_destroy_torch_distributed_process_group(
                    local.group
                )

        with ThreadPoolExecutor(DP) as pool:
            return list(pool.map(on_rank, range(DP)))

    return run


WIDTHS = [1, 2, 3, 1, 1, 2, 3, 3]


def test_a_decode_step_carries_every_rank_padded_to_the_rung(group):
    ranks = group([decode(w) for w in WIDTHS])

    # The largest batch is 3; the rung at or above it is 4, one query row each.
    assert [view.moe_rows for _, view in ranks] == [DP * 4 * 1] * DP


def test_a_mixed_prefill_step_carries_each_rank_its_own_count(group):
    batches = [prefill(40)] + [decode(w) for w in WIDTHS[1:]]

    ranks = group(batches)

    assert [view.moe_rows for _, view in ranks] == [40 + sum(WIDTHS[1:])] * DP


def test_the_group_price_and_rows_do_not_depend_on_rank_order(group):
    batches = [prefill(40), prefill(8)] + [decode(w) for w in WIDTHS[2:]]

    forward = group(batches)
    reverse = group(batches[::-1])

    assert {s for s, _ in forward} == {s for s, _ in reverse}
    assert len({s for s, _ in forward}) == 1
    assert {v.moe_rows for _, v in forward} == {v.moe_rows for _, v in reverse}
