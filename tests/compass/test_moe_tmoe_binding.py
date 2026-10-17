# SPDX-License-Identifier: MIT
"""`BatchView.moe_rows` and `draft_moe_rows` against the rows ATOM's own MoE
gather carries, on a two-rank gloo group.

Each rank settles its step with `ForwardMode.decide`, sets the forward context
`ModelRunner.prepare_model` sets from it, and runs `dp_gather_hidden_and_router`
as `FusedMoE.forward_impl_graph` calls it. An MTP draft's first pass re-points
that context with `Drafter._publish_draft_shape`, at the arguments
`EagleProposer.propose` passes on its first pass. Both are compiled from their
source files, because importing either module needs a driver; the DP group they
call is a gloo group behind the two methods the gather uses.
"""

import ast
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
from test_dp_step_max import prefill, runner
from test_moe_gather_rows import decode
from test_runner_non_allocating import REPO

import atom.utils.distributed.utils as dist_utils
import atom.utils.forward_context as fc
from atom.compass.runner import projection
from atom.utils import get_open_port

DP = 2
SET_CONTEXT = threading.Lock()  # `set_forward_context` writes one global


class Gloo:
    """aiter's DP group coordinator, as far as the gather calls it."""

    def __init__(self, group):
        self.cpu_group = group
        self.device_communicator = self

    def all_gather(self, x, use_custom=False, dim=0):
        parts = [torch.empty_like(x) for _ in range(DP)]
        dist.all_gather(parts, x, group=self.cpu_group)
        return torch.cat(parts, dim)

    def all_gatherv(self, x, dim=0, sizes=()):
        padded = x.new_zeros((max(sizes), *x.shape[1:]))
        padded[: len(x)] = x
        parts = [torch.empty_like(padded) for _ in sizes]
        dist.all_gather(parts, padded, group=self.cpu_group)
        return torch.cat([p[:n] for p, n in zip(parts, sizes)], dim)


def _source(path, names, cls=None):
    body = ast.parse(path.read_text()).body
    if cls is not None:
        body = next(n for n in body if getattr(n, "name", None) == cls).body
    return [ast.unparse(n) for n in body if getattr(n, "name", None) in names]


@pytest.fixture
def group(monkeypatch):
    """`run(batches, ladder)`: each rank settles and gathers its own batch."""
    monkeypatch.setattr(
        dist_utils, "_get_default_timeout", lambda _: timedelta(seconds=10)
    )
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    local = threading.local()

    def dp_group():
        return Gloo(local.group)

    monkeypatch.setitem(
        sys.modules, "aiter.dist.parallel_state", SimpleNamespace(get_dp_group=dp_group)
    )
    atom = {
        "torch": torch,
        "DPMetadata": fc.DPMetadata,
        "get_forward_context": fc.get_forward_context,
        "get_dp_group": dp_group,
    }
    gather = (
        "pad_for_all_gather",
        "all_gather_with_padding",
        "all_gatherv",
        "dp_gather_hidden_and_router",
    )
    for source in _source(REPO / "atom/model_ops/moe.py", gather) + _source(
        REPO / "atom/spec_decode/drafter.py", ("_publish_draft_shape",), "Drafter"
    ):
        exec(source, atom)  # noqa: S102

    def gathered(forward):
        """The rows `FusedMoE.forward_impl_graph` gathers under `forward`."""
        rows = torch.zeros(forward.context.running_tokens, 2)
        eager = not forward.context.running_tokens_are_unified
        hidden, *_ = atom["dp_gather_hidden_and_router"](
            rows, rows[:, :1], eager, forward, dp_group()
        )
        return hidden.shape[0]

    def on_rank(rank, batch, ladder, port):
        r = runner(None, dp_size=DP, dp_rank=rank)
        r.capture_sizes_np = np.array(ladder, dtype=np.int32)
        r.enforce_eager = False
        r.drafter = SimpleNamespace(is_block_drafter=False)
        config = SimpleNamespace(
            parallel_config=r.config.parallel_config,
            compilation_config=SimpleNamespace(static_forward_context={}),
        )
        local.group = dist_utils.stateless_init_torch_distributed_process_group(
            "127.0.0.1", port, rank, DP, backend="gloo"
        )
        try:
            mode = projection.forward_mode(batch, r)
            view = projection.batch_view(batch, mode, r)
            context = fc.Context(
                positions=None,
                is_prefill=mode.is_prefill,
                scheduled_bs=mode.scheduled_bs,
                scheduled_tokens=mode.scheduled_tokens,
                running_bs=mode.running_bs,
                running_tokens=mode.running_tokens,
                running_tokens_are_unified=mode.running_tokens_are_unified,
                forward_mode=mode,
            )
            with SET_CONTEXT:
                fc.set_forward_context(
                    attn_metadata=None,
                    atom_config=config,
                    context=context,
                    num_tokens=mode.scheduled_tokens,
                    num_tokens_across_dp=mode.sync.num_tokens_across_dp,
                )
                fc._forward_context_local.ctx = forward = fc.get_forward_context()
            target = gathered(forward)
            stream = mode.scheduled_tokens  # the target's tokens, unpadded
            atom["_publish_draft_shape"](
                SimpleNamespace(config=config),
                forward,
                scheduled_tokens=stream,
                running_tokens=stream,
                running_tokens_are_unified=False,
            )
            return (view.moe_rows, target), (view.draft_moe_rows, gathered(forward))
        finally:
            fc._forward_context_local.ctx = None
            dist_utils.stateless_destroy_torch_distributed_process_group(local.group)

    def run(batches, ladder):
        same = [ladder] * DP, [get_open_port()] * DP
        with ThreadPoolExecutor(DP) as pool:
            return list(pool.map(on_rank, range(DP), batches, *same))

    return run


def test_an_off_ladder_uniform_decode_pads_every_rank_to_the_group(group):
    # The group's largest batch, 3, is above the ladder, so it is the height.
    ranks = group([decode(3), decode(2)], ladder=[1, 2])

    assert ranks == [((6, 6), (5, 5))] * DP


def test_a_rank_prefilling_sends_each_rank_its_own_count(group):
    ranks = group([prefill(40), decode(2)], ladder=[1, 2, 4])

    assert ranks == [((42, 42), (42, 42))] * DP


def test_one_rank_states_no_gathered_rows():
    r = runner(None)
    r.drafter = SimpleNamespace(is_block_drafter=False)
    batch = decode(3)

    view = projection.batch_view(batch, projection.forward_mode(batch, r), r)

    assert (view.moe_rows, view.draft_moe_rows) == (None, None)
