# SPDX-License-Identifier: MIT
"""What a capture states for the other ranks of its DP and EP groups.

A capture runs one rank. Where ATOM reads its peers, two rules answer:

* Every peer scheduled what this rank did. The step's one DP all-gather
  (`sync_dp_metadata`) and the per-rank counts built from it
  (`DPMetadata.make`) run on the host with this rank's row in every slot.
* What the DP group gathers for the MoE, or MORI dispatches to this rank, is
  `moe_rows` rows, the group's width times this rank's; what comes back is
  this rank's rows.

Each exchange is one `declared` op named for ATOM's call (`dp.all_gatherv`,
`mori.dispatch`), its tensors the ones ATOM passed and got back.
"""

import types

from torch.utils import _pytree

from atom.compass import ir


def _declared(recorder, name, fn):
    """`fn` as one declared op `name`; what it dispatches inside is not ATOM's."""

    def call(*args, **kwargs):
        with recorder.quiet():
            out = fn(*args, **kwargs)
        ins, outs = _pytree.tree_leaves((args, kwargs)), _pytree.tree_leaves(out)
        recorder.add(name, ir.NodeKind.DECLARED, ins, outs)
        return out

    return call


def patches(recorder, own, moe_rows):
    """`(owner, name, value)` for each patch that answers for the DP peers."""
    import torch.distributed as dist
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    from atom.model_ops import moe
    from atom.utils.forward_context import DPMetadata
    from atom.utils.tbo import ubatching

    sync, make = ubatching.sync_dp_metadata, DPMetadata.make

    def on_host(fn, *args, **kwargs):
        with unset_fake_temporarily():
            return fn(*args, **kwargs)

    def every_peer_is_this_rank(gathered, local, group=None, async_op=False):
        for slot in gathered:
            slot.copy_(local)

    def sync_dp_metadata(**kwargs):
        gather, dist.all_gather = dist.all_gather, every_peer_is_this_rank
        try:
            return on_host(sync, **kwargs)
        finally:
            dist.all_gather = gather

    def rows(x, height):
        return x.new_empty((height, *x.shape[1:]))

    gathers = {
        "all_gatherv": lambda x, sizes, group: rows(x, moe_rows),
        "reduce_scatterv": lambda x, sizes, group: rows(x, own),
        "all_gather_with_padding": lambda x, use_cag=True: (rows(x, moe_rows), own),
        "reduce_scatter_with_unpadding": lambda x, local_tokens: rows(x, own),
    }
    return [
        (ubatching, "sync_dp_metadata", sync_dp_metadata),
        (DPMetadata, "make", staticmethod(lambda *a, **k: on_host(make, *a, **k))),
        *((moe, n, _declared(recorder, f"dp.{n}", f)) for n, f in gathers.items()),
    ]


def install_mori(recorder, moe_rows):
    """Give the EP group an all-to-all manager whose MORI handles are declared."""
    from aiter.dist.parallel_state import get_ep_group

    group = get_ep_group()

    class Handle:
        config = types.SimpleNamespace(
            kernel_type=types.SimpleNamespace(name="IntraNode")
        )

        def __init__(self, args):
            self.capacity = args["max_num_tokens_per_dp_rank"]
            self.dispatch = _declared(recorder, "mori.dispatch", self._dispatch)
            self.combine = _declared(recorder, "mori.combine", self._combine)

        def _dispatch(self, x, weights, scales, ids, **kwargs):
            n, k = moe_rows, ids.shape[1]
            return (
                x.new_empty((n, x.shape[1])),
                weights.new_empty((n, k)),
                None if scales is None else scales.new_empty((n, scales.shape[1])),
                ids.new_empty((n, k)),
                ids.new_empty((1,)),
            )

        def _combine(self, x, weights, ids, **kwargs):
            return x.new_empty((self.capacity, x.shape[-1])), None

    manager = types.SimpleNamespace(
        rank=group.rank_in_group,
        world_size=group.world_size,
        internode=False,
        get_handle=lambda args, index=0: Handle(args),
    )
    group.device_communicator = types.SimpleNamespace(
        all2all_manager=manager, destroy=lambda: None
    )
