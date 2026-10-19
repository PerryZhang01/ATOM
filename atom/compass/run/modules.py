# SPDX-License-Identifier: MIT
"""One rank's meta build, read as the modules the step roofline prices.

`walk(tree)` goes down the tree from its root by ATOM module class, matched by
name so the walk imports none of them, and hands
back one `Module` per module that works on a step, its dims read off the
module's own per-rank parameter shapes and attributes:

- a linear class (`LinearBase`, `torch.nn.Linear`): a GEMM of its weight,
  joining an all-reduce over its TP group when it reduces its result. A
  weight that is not a matrix is the kernel of a depthwise causal convolution
  (GDN keeps its `conv1d` in a column-parallel linear): one pass over its
  channels.
- `VocabParallelEmbedding`, `ParallelLMHead`: a lookup and its all-reduce; a
  GEMM and its all-gather.
- `RMSNorm`, `GemmaRMSNorm`, `RMSNormGated`: one pass over the weight's
  width; an `RMSNorm` built with `fused_allreduce` joins its all-reduce
  inside the norm.
- `aiter`'s `RotaryEmbedding`: one pass over `rotary_dim`, once for every
  module that holds it.
- `Attention` (not MLA), `LinearAttention`: attention over the step's rows,
  at their own heads and dims; neither is gone into.
- `SiluAndMul`: the input width of the one row-parallel linear beside it.
- a torch or transformers activation module, the vision tower, and V4's
  rotary tables: nothing. ATOM's text stacks call no bare activation module
  (GDN holds one it never calls), the backend refuses a step with images, and
  V4 reads its tables inside the attention and compressor kernels.

DeepSeek-V4's modules, each priced and then gone into:

- `Block`: two hyper-connection mixes, one per sub-layer, each folding the
  previous sub-layer's output into the widened residual and reducing it to the
  next input; the block's two norms run inside them.
- `DeepseekV4Attention`: sparse attention over the sliding window and the
  compressed KV, with the indexer's scoring where the layer has one. Its
  linears and the compressors' are priced as linears.
- `Expert` (the shared expert): its activation, between its two linears.
- `FusedMoE`, not gone into: the routed experts, at its quant method's
  activation format, and the exchange around them: the data-parallel group's
  gather and reduce-scatter, MoRI's dispatch and combine under expert
  parallelism, at the formats MoRI sends, or the tensor-parallel all-reduce
  on one data-parallel rank. An expert-parallel MoE over a data-parallel
  group whose exchange is not MoRI's (`--all2all-backend rccl` or `none`,
  `ATOM_DISABLE_MORI_EP`), and the Mega kernel, are unclassified.

A module of any other class is gone into when it has children. One without,
or a listed one in a format or layout not priced here, is unclassified: the
first of each class comes back as an `unclassified` module named by class
and path, which the backend refuses. Each module carries the index of the
decoder layer it sits in, read off its path.
"""

import dataclasses
import re

import torch

from atom.compass.backends.step_roofline import UNCLASSIFIED, Module

#: Weight, KV and state formats a module is priced at, by dtype.
FORMATS = {
    torch.bfloat16: "bf16",
    torch.float8_e4m3fn: "fp8",
    torch.float8_e4m3fnuz: "fp8",
    torch.float4_e2m1fn_x2: "fp4",
    "bf16": "bf16",
    "fp8": "fp8",
}
#: A module that prices nothing; one to go into.
NOTHING, DESCEND = object(), object()
OPS = "atom.model_ops."
V4 = "atom.models.deepseek_v4."
ROW = OPS + "linear.RowParallelLinear"
NORMS = tuple(
    OPS + "layernorm." + n for n in ("RMSNorm", "GemmaRMSNorm", "RMSNormGated")
)
#: Modules that price nothing: activation modules ATOM's text stacks hold and
#: never call, and the vision tower.
SILENT = ("torch.nn.modules.activation", "transformers.activations")
VISION = "atom.models.qwen3_5_vl.Qwen3VisionTransformer"
#: V4's rotary tables, read inside the attention and compressor kernels.
V4_ROPE = V4 + "_V4RoPE"
#: Linear quant types whose activations stay bf16 into the matrix product.
UNQUANTIZED_INPUT = ("No", "per_1x32")
LAYER = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def walk(tree: torch.nn.Module) -> list[Module]:
    """The modules of `tree` the step roofline prices, in visiting order."""
    found: list[Module] = []
    seen: set[type] = set()

    def visit(module, name, parent):
        m = _classify(module, name, parent)
        *priced, then = m if isinstance(m, tuple) else (m,)
        if isinstance(then, Module):
            priced, then = [then], NOTHING
        layer = LAYER.search(name)
        layer = int(layer.group(1)) if layer else -1
        found.extend(dataclasses.replace(p, layer=layer) for p in priced)
        if then is DESCEND:
            for child, sub in module.named_children():
                visit(sub, f"{name}.{child}" if name else child, module)
        elif then is None and type(module) not in seen:
            seen.add(type(module))
            where = f"{type(module).__name__} at {name or 'the root'}"
            found.append(Module(where, UNCLASSIFIED))

    visit(tree, "", None)
    return found


def _is(module, *names: str) -> bool:
    """Whether `module`'s class is, or derives from, a class of `names`."""
    return any(
        f"{c.__module__}.{c.__qualname__}" in names for c in type(module).__mro__
    )


def _classify(module, name, parent):
    """`module`'s `Module`, `NOTHING`, `DESCEND`, or None if unclassified."""
    weight = getattr(module, "weight", None)
    fmt = FORMATS.get(getattr(weight, "dtype", None))
    if _is(module, V4 + "Block"):
        mix = Module(name, "mhc", (module.hc_mult, module.attn_norm.weight.shape[0]))
        return mix, mix, DESCEND
    if _is(module, V4 + "DeepseekV4Attention"):
        ix = module.indexer
        dims = (module.n_local_heads, module.head_dim, module.window_size)
        dims += (module.compress_ratio,)
        dims += (ix.index_topk, ix.n_heads, ix.head_dim) if ix else (0, 0, 0)
        # ponytail: the KV is read at fp8, its narrowest format; the window
        # rows held in bf16 are left out of this lower bound.
        return Module(name, "sparse_attention", dims, "fp8"), DESCEND
    if _is(module, V4 + "Expert"):
        return Module(name, "act", (module.w2.weight.shape[1],)), DESCEND
    if _is(module, OPS + "moe.FusedMoE"):
        return _moe(module, name, parent)
    if _is(module, OPS + "embed_head.ParallelLMHead"):
        return fmt and Module(name, "lm_head", tuple(weight.shape), fmt, module.tp_size)
    if _is(module, OPS + "embed_head.VocabParallelEmbedding"):
        return Module(name, "embedding", (weight.shape[1],), ranks=module.tp_size)
    if _is(module, OPS + "linear.LinearBase", "torch.nn.modules.linear.Linear"):
        if not fmt:
            return None
        if weight.dim() != 2:
            return Module(name, "conv", (weight.shape[0],), fmt)
        reduces = _is(module, ROW) and module.reduce_results
        ranks = module.tp_size if reduces else 1
        quant = getattr(getattr(module, "quant_type", None), "name", "No")
        act = "bf16" if quant in UNQUANTIZED_INPUT else fmt
        return Module(name, "gemm", tuple(weight.shape), fmt, ranks, act=act)
    if _is(module, *NORMS):
        if parent is not None and _is(parent, V4 + "Block"):
            return NOTHING
        fused = getattr(module, "fused_allreduce", False) and module.tp_size > 1
        ranks = module.tp_size if fused else 1
        return Module(name, "norm", (weight.shape[0],), ranks=ranks, fused=fused)
    if _is(module, "aiter.rotary_embedding.RotaryEmbedding"):
        return Module(name, "rotary", (module.rotary_dim,))
    if _is(module, OPS + "paged_attention.Attention"):
        kv = FORMATS.get(module.kv_cache_dtype)
        dims = (module.num_heads, module.num_kv_heads, module.head_dim)
        return kv and not module.use_mla and Module(name, "attention", dims, kv) or None
    if _is(module, OPS + "base_attention.LinearAttention"):
        tp, state = module.impl.tp_size, FORMATS.get(module.A_log.dtype)
        dims = (module.num_v_heads // tp, module.num_k_heads // tp)
        dims += (module.head_k_dim, module.head_v_dim)
        return state and Module(name, "linear_attention", dims, state)
    if _is(module, OPS + "activation.SiluAndMul"):
        down = [c for c in parent.children() if _is(c, ROW)]
        return (
            len(down) == 1 and Module(name, "act", (down[0].weight.shape[1],)) or None
        )
    if _is(module, VISION, V4_ROPE) or type(module).__module__.startswith(SILENT):
        return NOTHING
    return DESCEND if next(module.children(), None) is not None else None


def _moe(module, name, parent):
    """The routed experts and their exchange, or None if unclassified.

    `dims` are (experts, local experts, top k, intermediate per rank, hidden,
    group ranks, data-parallel ranks, MoRI's split send and receive). The
    data-parallel width is the MoE group's over the attention's TP group,
    which the parent MoE block holds. Under expert parallelism over that
    group, `wire` is what MoRI's dispatch sends, as ATOM resolves it, and
    what its combine sends back.
    """
    method, w13 = module.quant_method, module.w13_weight
    act = getattr(getattr(method, "act_quant", None), "value", None)
    if getattr(method, "fp8_activations", False):
        act = "fp8"
    fmt, attention_tp = FORMATS.get(w13.dtype), getattr(parent, "tp_size", None)
    if not (act and fmt and attention_tp):
        return None
    config = module.moe_parallel_config
    ranks = config.ep_size if config.use_ep else config.tp_size
    split = int(config.use_ep and config.low_latency)
    dims = (module.global_num_experts, w13.shape[0], module.top_k)
    dims += (w13.shape[1] // 2, module.hidden_size, ranks, ranks // attention_tp, split)
    fused = module._comm_fused_moe is not None
    if _is(method, OPS + "moe.MegaMxfp4MoEMethod"):
        return None
    wire = ("bf16", "bf16")
    if config.use_ep and dims[6] > 1:
        from atom.model_ops.fused_moe.mori_prepare_finalize import (
            resolve_mori_dispatch,
        )
        from atom.utils import envs

        mori = config.requested_all2all_backend in ("auto", "mori")
        mori = mori and not envs.ATOM_DISABLE_MORI_EP
        sent = resolve_mori_dispatch(module.moe_config.in_dtype, module.hidden_size)
        sent = FORMATS.get(sent.dtype)
        if not (mori and sent):
            return None
        wire = (sent, "bf16" if envs.ATOM_MORI_COMBINE_QUANT == "none" else "fp8")
    return Module(name, "moe", dims, fmt, ranks, fused, act=act, wire=wire)
