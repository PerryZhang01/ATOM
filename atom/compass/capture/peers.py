# SPDX-License-Identifier: MIT
"""What a capture states for the other ranks of its DP and EP groups.

A capture runs one rank. Where ATOM reads its peers, two rules answer:

* Every peer scheduled what this rank did. The step's one DP all-gather
  (`sync_dp_metadata`), the per-rank counts built from it
  and the sizes read off them (`DPMetadata.make`, `get_sizes_across_dp`)
  run on the host with this rank's row in every slot.
* What the DP group gathers for the MoE, or MORI dispatches to this rank, is
  every rank's rows, the group's width times this rank's: for MORI, the height
  ATOM trims its buffer to, `T_moe`. The DP reduce-scatter gives back this
  rank's rows; MORI's combine gives back its buffer of
  `max_num_tokens_per_dp_rank` rows, which ATOM's finalize slices.

Each exchange is one `declared` op named `module:qualname` for the function
ATOM calls (`atom.model_ops.moe:all_gatherv`,
`mori.ops.dispatch_combine:EpDispatchCombineOp.dispatch`), its arguments the
ones ATOM passed by that function's parameter names, the group aside.
"""

import inspect
import types

import torch
from torch.utils import _pytree

from atom.compass import ir

#: MORI's dispatch-and-combine handle, whose two calls the EP exchanges are.
MORI = "mori.ops.dispatch_combine:EpDispatchCombineOp"


def _declared(recorder, name, fn):
    """`fn` as one declared op `name`; what it dispatches inside is not ATOM's."""
    from .forward import _atom_frames

    signature = inspect.signature(fn)

    def call(*args, **kwargs):
        with recorder.quiet():
            out = fn(*args, **kwargs)
        if recorder.on:
            slots = signature.bind(*args, **kwargs).arguments
            slots.update(slots.pop("kwargs", {}))
            slots.pop("group", None)
            outs = [t for t in _pytree.tree_leaves(out) if isinstance(t, torch.Tensor)]
            recorder.add(name, ir.NodeKind.DECLARED, slots, outs, None, _atom_frames())
        return out

    return call


def patches(recorder):
    """`(owner, name, value)` for each patch that answers for the DP peers."""
    import torch.distributed as dist
    from aiter.dist.parallel_state import get_dp_group
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    from atom.model_ops import moe
    from atom.utils.forward_context import DPMetadata
    from atom.utils.tbo import ubatching

    sync, make = ubatching.sync_dp_metadata, DPMetadata.make
    sizes = DPMetadata.get_sizes_across_dp
    dp = get_dp_group().world_size

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

    # `sizes` holds this rank's count in every slot.
    def all_gatherv(x, sizes, group):
        return rows(x, sum(sizes))

    def reduce_scatterv(x, sizes, group):
        return rows(x, sizes[0])

    def all_gather_with_padding(x, use_cag=True):
        return rows(x, dp * x.shape[0]), x.shape[0]

    def reduce_scatter_with_unpadding(x, local_tokens):
        return rows(x, local_tokens)

    gathers = (
        all_gatherv,
        reduce_scatterv,
        all_gather_with_padding,
        reduce_scatter_with_unpadding,
    )
    return [
        (ubatching, "sync_dp_metadata", sync_dp_metadata),
        (DPMetadata, "make", staticmethod(lambda *a, **k: on_host(make, *a, **k))),
        (DPMetadata, "get_sizes_across_dp", lambda self: on_host(sizes, self)),
        *(
            (moe, f.__name__, _declared(recorder, f"{moe.__name__}:{f.__name__}", f))
            for f in gathers
        ),
    ]


def install_mori(recorder):
    """Give the EP group an all-to-all manager whose MORI handles are declared."""
    from aiter.dist.parallel_state import get_ep_group

    group = get_ep_group()

    class Handle:
        """MORI's `EpDispatchCombineOp`, its two calls by its parameter names."""

        config = types.SimpleNamespace(
            kernel_type=types.SimpleNamespace(name="IntraNode")
        )

        def __init__(self, args):
            self.capacity = args["max_num_tokens_per_dp_rank"]
            self.dispatch = _declared(recorder, f"{MORI}.dispatch", self._dispatch)
            self.combine = _declared(recorder, f"{MORI}.combine", self._combine)

        def _dispatch(self, input, weights, scales, indices, **kwargs):
            n, k = group.world_size * input.shape[0], indices.shape[1]
            return (
                input.new_empty((n, input.shape[1])),
                weights.new_empty((n, k)),
                None if scales is None else scales.new_empty((n, scales.shape[1])),
                indices.new_empty((n, k)),
                indices.new_empty((1,)),
            )

        def _combine(self, input, weights, indices, **kwargs):
            return input.new_empty((self.capacity, input.shape[-1])), None

    manager = types.SimpleNamespace(
        rank=group.rank_in_group,
        world_size=group.world_size,
        internode=False,
        get_handle=lambda args, index=0: Handle(args),
    )
    group.device_communicator = types.SimpleNamespace(
        all2all_manager=manager, destroy=lambda: None
    )
