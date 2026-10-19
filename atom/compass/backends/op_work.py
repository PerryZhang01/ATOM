# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""A captured leaf's work, by op kind: the op-level extractor registry.

A leaf is one operator of an `op_graph` as its JSON holds it: `op`, its kind;
`args` by schema slot, a tensor as its shape, dtype, stride and role; `out`,
its output shapes. `work(leaf, view, tp, kv_format)` is the leaf's `Work` on a
step of `view`'s rows: F and C from the primitives its kind's extractor emits,
M the leaf's own tensor arguments read once and its outputs written once, plus
the ambient state it reads (the KV context in `kv_format`, the recurrent
state). A fused leaf is
one leaf, so its intermediates cost nothing. A tensor counts its distinct
elements, a broadcast dimension (stride 0) once. The leaf's own tensors are
priced at the shape it was captured at, padding as executed; the ambient state
at the view's rows.

The work reads nothing `bench.ops.signature` leaves out of a leaf's identity,
so one call prices every instance of a signature at a view.

Aten operators share one extractor, whose F is `torch.utils.flop_counter`'s
formula where it has one. A leaf that launches no kernel (a view, an
allocation, a wait) has no work: None. A collective spans the `tp` ranks of
the tensor-parallel group unless the leaf states its group size. A kind with
no extractor, or a leaf its extractor cannot price, is refused by name.
"""

from __future__ import annotations

import math

import torch
from torch.utils._pytree import tree_leaves
from torch.utils.flop_counter import flop_registry

from atom.compass.backends.provenance import Refusal
from atom.compass.backends.resolver import CostRefused
from atom.compass.backends.shape import BatchView
from atom.compass.backends.step_roofline import Module
from atom.compass.backends.step_roofline import _work as module_work
from atom.compass.backends.work import (
    Collective,
    Work,
    allgather,
    allreduce,
    attention,
    gemm,
)

SOURCE = "op-work"
#: The work primitives' operand formats, by dtype.
FORMATS = {
    "torch.float32": "fp32",
    "torch.bfloat16": "bf16",
    "torch.float8_e4m3fn": "fp8",
    "torch.float8_e4m3fnuz": "fp8",
}
#: Aten operators that allocate and launch nothing.
ALLOCATIONS = ("aten.empty.", "aten.empty_like.")


class _Unpriced(Exception):
    """A leaf its kind's extractor cannot price; `work` refuses it."""


def _fmt(dtype: str) -> str:
    if dtype not in FORMATS:
        raise _Unpriced(f"no operand format for {dtype}; known: {', '.join(FORMATS)}")
    return FORMATS[dtype]


def _size(dtype: str) -> int:
    return getattr(torch, dtype.removeprefix("torch.")).itemsize


def _tensors(value):
    if isinstance(value, dict) and "dtype" in value:
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _tensors(item)


def _bytes(tensor) -> int:
    shape, stride = tensor["shape"], tensor["stride"]
    return math.prod(n for n, s in zip(shape, stride) if s) * _size(tensor["dtype"])


def _out_dtype(op) -> str | None:
    """The outputs' dtype: the `dtype` slot's, else the first tensor's."""
    dtype = op["args"].get("dtype")
    if isinstance(dtype, str):
        return dtype
    return next((t["dtype"] for t in _tensors(list(op["args"].values()))), None)


def _boundary(op, written=(), outs=None) -> Work:
    """Each tensor argument read once, the `written` slots written once more,
    and each output written once in its dtype in `outs`, by default
    `_out_dtype`'s."""
    args = op["args"]
    moved = sum(map(_bytes, _tensors(list(args.values()))))
    moved += sum(_bytes(t) for slot in written for t in _tensors(args[slot]))
    if outs is None:
        outs = [_out_dtype(op)] * len(op["out"])
    moved += sum(math.prod(s) * _size(d) for s, d in zip(op["out"], outs))
    return Work({}, moved)


def _gemm(op, view, tp, kv) -> Work:
    """`A [.., k] @ B [n, k]^T` in `otype`, and the bias read once."""
    a, b, args = op["args"]["A"], op["args"]["B"], op["args"]
    *rows, k = a["shape"]
    out = _fmt(args.get("otype") or a["dtype"])
    bias = sum(map(_bytes, _tensors(args.get("bias"))))
    work = gemm(
        math.prod(rows), b["shape"][0], k, _fmt(a["dtype"]), _fmt(b["dtype"]), out
    )
    return work + Work({}, bias)


def _attention(op, view, tp, kv) -> Work:
    """Causal attention over the view's rows: Q read and O written, each row's
    K and V over its context read once in the KV cache's format `kv`, the new
    tokens' among them. F and M take heads times head dim only, so each side
    counts as one head of its width."""
    q, k = op["args"]["q"], op["args"]["k"]
    rows = view.requests
    return attention(
        [r.query_tokens for r in rows],
        [r.context_tokens for r in rows],
        math.prod(q["shape"][1:]),
        1,
        math.prod(k["shape"][1:]),
        _fmt(q["dtype"]),
        kv,
    )


def _linear_attention(op, view, tp, kv) -> Work:
    """GDN's recurrence over the view's rows, as the step roofline counts it:
    value heads and width off `core_attn_out`, key heads off `mixed_qkv`, the
    state in the activations' format, where ATOM keeps it for these models."""
    qkv, out = op["args"]["mixed_qkv"], op["args"]["core_attn_out"]
    *_, heads, dv = out["shape"]
    # ponytail: the key width is taken equal to the value width, which the
    # leaf does not state apart; the module walk reads it off the layer.
    key_heads = (qkv["shape"][-1] - heads * dv) // (2 * dv)
    dims = (heads, key_heads, dv, dv)
    return module_work(
        Module("", "linear_attention", dims, _fmt(qkv["dtype"])), view, 0, 0
    )


def _gather(table):
    """A lookup of `out`'s rows in the `table` slot: the rows it writes and
    the indices. The rows it reads are not counted: a masked lookup reads only
    its shard's ids, and a repeated id reads its row once."""

    def extract(op, view, tp, kv) -> Work:
        args = op["args"]
        rows = sum(math.prod(s) for s in op["out"]) * _size(args[table]["dtype"])
        rest = [v for slot, v in args.items() if slot != table]
        return Work({}, sum(map(_bytes, _tensors(rest))) + rows)

    return extract


def _in_place(*slots):
    return lambda op, view, tp, kv: _boundary(op, slots)


def _all_reduce(op, view, tp, kv) -> Work:
    return _boundary(op) + allreduce(tp, _bytes(op["args"]["tensor"]))


def _all_gather(op, view, tp, kv) -> Work:
    gathered = math.prod(op["out"][0]) * _size(op["args"]["input"]["dtype"])
    return _boundary(op) + allgather(op["args"]["group_size"], gathered)


def _broadcast(op, view, tp, kv) -> Work:
    """The source sends the whole buffer at least once."""
    sent = _bytes(op["args"]["input"])
    return _boundary(op) + Work(comm=(Collective(tp, sent),) if tp > 1 else ())


def _kv_indices(op, view, tp, kv) -> Work:
    # ponytail: the block-table entries it reads and the indices it writes
    # scale with each row's blocks, which the leaf does not state; only the
    # row offsets are counted. Add them when the KV block size reaches here.
    return Work({}, _bytes(op["args"]["kv_indptr_convert"]))


def _shape(value):
    if isinstance(value, dict) and "dtype" in value:
        return value["shape"]
    return [_shape(v) for v in value] if isinstance(value, list) else value


def _meta(slot, value):
    """`value` as a call on the meta device takes it: a tensor as an empty
    meta tensor of its shape, dtype and stride."""
    if isinstance(value, list):
        return [_meta(slot, v) for v in value]
    if isinstance(value, dict) and "dtype" in value:
        dtype = getattr(torch, value["dtype"].removeprefix("torch."))
        return torch.empty_strided(
            value["shape"], value["stride"], dtype=dtype, device="meta"
        )
    if slot == "device":
        return "meta"
    if isinstance(value, str) and value.startswith("torch."):
        return getattr(torch, value.removeprefix("torch."), value)
    return value


def _aten(op, view, tp, kv) -> Work | None:
    """Any aten operator: its boundary, each output in the dtype a meta-device
    call of it returns, and torch's flop formula when it has one. A view,
    which aliases its input, writes nothing and keeps the dtype, launches no
    kernel; nor does an allocation."""
    name, args = op["op"], op["args"]
    if name.startswith(ALLOCATIONS):
        return None
    _, packet, overload = name.split(".")
    target = getattr(getattr(torch.ops.aten, packet), overload)
    try:
        result = target(**{slot: _meta(slot, v) for slot, v in args.items()})
    except Exception as error:  # noqa: BLE001 - refused with what stopped it
        raise _Unpriced(f"a meta-device call raised {error!r}") from None
    outs = [str(t.dtype) for t in tree_leaves(result) if isinstance(t, torch.Tensor)]
    schema = target._schema
    first = next(_tensors(list(args.values())), None)
    aliased = all(r.alias_info for r in schema.returns)
    writes = any(a.alias_info and a.alias_info.is_write for a in schema.arguments)
    if aliased and not writes and first and set(outs) <= {first["dtype"]}:
        return None
    # An output that is an argument it writes is not written twice.
    rewritten = any(r.alias_info and r.alias_info.is_write for r in schema.returns)
    work = _boundary(op, outs=[] if rewritten else outs)
    formula = flop_registry.get(target.overloadpacket)
    if formula is None:
        return work
    out = op["out"][0] if len(op["out"]) == 1 else op["out"]
    flops = formula(*map(_shape, args.values()), out_val=out)
    return work + Work({_fmt(first["dtype"]): flops})


#: The extractor of each op kind but aten's, by the name a leaf carries.
EXTRACTORS = {
    "aiter.gemm_a16w16.default": _gemm,
    "aiter.unified_attention_with_output_base.default": _attention,
    "aiter.linear_attention_with_output_base.default": _linear_attention,
    "aiter.silu_and_mul.default": _in_place(),
    "aiter._fused_qk_rmsnorm_group_quant_kernel.default": _in_place(),
    "aiter.mixed_sample_outer_exponential.default": _in_place(),
    "aiter.masked_embedding.default": _gather("weight"),
    "aten.embedding.default": _gather("weight"),
    "aten.index.Tensor": _gather("self"),
    "aiter.all_reduce_.default": _all_reduce,
    "_c10d_functional.all_gather_into_tensor.default": _all_gather,
    "_c10d_functional.broadcast.default": _broadcast,
    "_c10d_functional.wait_tensor.default": lambda op, view, tp, kv: None,
    "atom.model_ops.layernorm:fused_qk_norm": _in_place("q", "k"),
    "atom.model_ops.triton_mrope:try_mrope_qk_fused": _in_place("q", "k"),
    "atom.model_ops.sampler:_apply_greedy_tokens": _in_place(),
    "atom.model_ops.decode_input_ids:fill_deferred_decode_ids": _in_place(),
    "atom.utils.block_convert:kv_indices_generate_triton": _kv_indices,
}


def work(op: dict, view: BatchView, tp: int, kv_format: str) -> Work | None:
    """`op`'s work on a step of `view`'s rows in a TP group of `tp` ranks
    whose KV cache holds `kv_format` (a work format, "bf16" or "fp8"), or None
    when it launches no kernel. A kind with no extractor, or a leaf its
    extractor cannot price, is refused."""
    name = op["op"]
    extract = EXTRACTORS.get(name) or (name.startswith("aten.") and _aten)
    if not extract:
        reason = f"no extractor for the op kind {name}"
        raise CostRefused(view, [Refusal(SOURCE, reason)])
    try:
        return extract(op, view, tp, kv_format)
    except _Unpriced as error:
        reason = f"the op kind {name}: {error}"
    raise CostRefused(view, [Refusal(SOURCE, reason)])
