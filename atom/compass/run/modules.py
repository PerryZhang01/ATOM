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
- a torch or transformers activation module, and the vision tower: nothing.
  ATOM's text stacks call no bare activation module (GDN holds one it never
  calls), and the backend refuses a step with images.

A module of any other class is gone into when it has children. One without,
or a listed one in a format or layout not priced here, is unclassified: the
first of each class comes back as an `unclassified` module named by class
and path, which the backend refuses.
"""

import torch

from atom.compass.backends.step_roofline import UNCLASSIFIED, Module

#: Weight, KV and state formats a module is priced at, by dtype.
FORMATS = {
    torch.bfloat16: "bf16",
    torch.float8_e4m3fn: "fp8",
    torch.float8_e4m3fnuz: "fp8",
    "bf16": "bf16",
    "fp8": "fp8",
}
#: A module that prices nothing; one to go into.
NOTHING, DESCEND = object(), object()
OPS = "atom.model_ops."
ROW = OPS + "linear.RowParallelLinear"
NORMS = tuple(
    OPS + "layernorm." + n for n in ("RMSNorm", "GemmaRMSNorm", "RMSNormGated")
)
#: Modules that price nothing: activation modules ATOM's text stacks hold and
#: never call, and the vision tower.
SILENT = ("torch.nn.modules.activation", "transformers.activations")
VISION = "atom.models.qwen3_5_vl.Qwen3VisionTransformer"


def walk(tree: torch.nn.Module) -> list[Module]:
    """The modules of `tree` the step roofline prices, in visiting order."""
    found: list[Module] = []
    seen: set[type] = set()

    def visit(module, name, parent):
        m = _classify(module, name, parent)
        if m is DESCEND:
            for child, sub in module.named_children():
                visit(sub, f"{name}.{child}" if name else child, module)
        elif m is None:
            if type(module) not in seen:
                seen.add(type(module))
                where = f"{type(module).__name__} at {name or 'the root'}"
                found.append(Module(where, UNCLASSIFIED))
        elif m is not NOTHING:
            found.append(m)

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
        return Module(name, "gemm", tuple(weight.shape), fmt, ranks)
    if _is(module, *NORMS):
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
    if _is(module, VISION) or type(module).__module__.startswith(SILENT):
        return NOTHING
    return DESCEND if next(module.children(), None) is not None else None
