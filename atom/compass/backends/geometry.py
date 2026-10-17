# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The KV shape a stand-in model presents, and how parallelism divides it.

This is a test double. From the engine's side it is indistinguishable from a
model in the one respect the engine needs before it can start -- it says how
many bytes a KV block costs -- and it is not an accurate account of any real
one. It omits the fp32 scale plane the aiter backends carry, and it sizes no
per-request recurrent state. What it buys is that everything downstream is
then real: ATOM's own `page_pool`, `plan_pools` and `BlockManager` do the
sizing, and they get a genuinely different answer per parallel width rather
than a number this module decided.

Nothing here imports the engine, so the arithmetic runs anywhere Python does.
The two places the engine owns the answer are handed in rather than guessed:
which layers a pipeline stage holds comes from ATOM's own partitioner as a
`layer_range`, and the byte budget is the caller's.

What each axis does to the KV, and the one that surprises people:

- **TP** cuts the KV heads, bounded by grouped-query attention. Below one head
  per rank the heads are replicated instead of cut, so a rank's KV stops
  shrinking -- a model with four KV heads holds the same KV at eight ranks as
  at four. The bound is read off the config's KV head count, never inferred
  from the query heads.
- **PP** gives a stage its own layers, so its KV is that slice and its block
  is that much cheaper. It is also what makes more than one participant for
  the clock to coordinate.
- **DP** replicates: each engine holds a full KV, so the per-engine block is
  unchanged and there are `dp_size` of them.
- **EP** shards experts and leaves the KV alone -- structurally so here, since
  the expert width is not an input to the geometry at all.

The collective that rides on that last point is measured, not deduced. With
one data-parallel rank ATOM builds no all-to-all at all, even with expert
parallelism on: the MoE is local compute masked per rank plus the
tensor-parallel all-reduce. So `collectives()` names one on the data-parallel
width and never on the expert width, which is the way round that gets that
case wrong.

`collectives()` is a **necessary** condition, not a sufficient one, and the
gap is named here because nothing downstream can see it. ATOM builds its
peer-to-peer MoE path only when two further things hold that these widths
cannot express: the deployment is not being simulated wider than the box, and
the peer-to-peer library is installed. There is also one case in the other
direction -- the mega MoE backend installs its experts unconditionally and
does run peer-to-peer at a single data-parallel rank. So a reader may take a
collective missing from this list as absent, and must not take one present in
it as built.
"""

from __future__ import annotations

from dataclasses import dataclass

# Element sizes, under the spellings a HF config and ATOM's --kv-cache-dtype
# each use for the same type.
_DTYPE_BYTES = {
    "float32": 4,
    "fp32": 4,
    "float16": 2,
    "fp16": 2,
    "half": 2,
    "bfloat16": 2,
    "bf16": 2,
    "float8_e4m3fn": 1,
    "float8_e4m3fnuz": 1,
    "fp8": 1,
    "int8": 1,
    "uint8": 1,
}

# Layer kinds that hold a cache of every past token, which is the only thing
# this geometry knows how to size.
_PAGED_LAYER_KINDS = frozenset({"full_attention"})

# Layer kinds that hold no such cache, keeping a per-request recurrent state
# instead. A block costs nothing for them.
_UNCACHED_LAYER_KINDS = frozenset({"linear_attention", "mamba", "recurrent"})

# Fields ATOM reads to recognise a stack whose layers are not all alike. One
# of these on a config that names no layer kinds means the kinds are missing,
# not that the stack is uniform.
_MIXED_STACK_FIELDS = (
    "full_attention_interval",
    "hybrid_layer_pattern",
    "linear_attn_config",
)


def dtype_bytes(dtype) -> int:
    """Element size of a dtype given as a name or carried as a torch dtype."""
    size = getattr(dtype, "itemsize", None)
    if size is not None:
        return int(size)
    name = str(dtype).rsplit(".", 1)[-1].lower()
    if name not in _DTYPE_BYTES:
        raise ValueError(
            f"no element size known for KV dtype {dtype!r}; name it as one of "
            f"{', '.join(sorted(_DTYPE_BYTES))}"
        )
    return _DTYPE_BYTES[name]


def paged_layers(hf_config, start: int, end: int) -> int:
    """How many layers of a half-open span hold a cache of every past token.

    A config that names its layer kinds is counted by them, and a kind this
    module does not recognise is refused rather than taken for ordinary
    attention. A windowed or chunked kind is the case that makes the
    distinction matter: it keeps a bounded cache, sized by the window rather
    than by the history, and counting it here would add a full history's bytes
    to every block.

    A config that names no kinds is a uniform stack, and that claim is
    checked rather than assumed. On a stack that is one full-attention layer
    in four, reading it as uniform gives four times the bytes per block and a
    quarter of the blocks -- and the engine starts on the result.
    """
    kinds = getattr(hf_config, "layer_types", None)
    if kinds is None:
        mixed = [
            field
            for field in _MIXED_STACK_FIELDS
            if getattr(hf_config, field, None) is not None
        ]
        if mixed:
            raise ValueError(
                f"this config carries {', '.join(mixed)} but no layer_types, so which "
                "of its layers hold a cache is not stated; ATOM's own config classes "
                "fill layer_types in, and building the config through one settles it"
            )
        return end - start
    span = kinds[start:end]
    known = _PAGED_LAYER_KINDS | _UNCACHED_LAYER_KINDS
    unknown = sorted({kind for kind in span if kind not in known})
    if unknown:
        raise ValueError(
            f"unrecognised layer kind(s) {', '.join(unknown)}; this geometry sizes a "
            f"cache of the whole history ({', '.join(sorted(_PAGED_LAYER_KINDS))}) or "
            f"none at all ({', '.join(sorted(_UNCACHED_LAYER_KINDS))}), and a bounded "
            "cache is neither"
        )
    return sum(1 for kind in span if kind in _PAGED_LAYER_KINDS)


def kv_heads_per_rank(total_kv_heads: int, tp_size: int) -> int:
    """How many KV heads one tensor-parallel rank holds.

    Either the heads divide across the ranks, or the ranks divide across the
    heads and every rank keeps a replica of one. Any other width is refused:
    ATOM's own sharding asserts the same two cases, so a rounded answer here
    would size a pool the engine then declines to build.
    """
    if total_kv_heads < 1 or tp_size < 1:
        raise ValueError(f"need positive widths, got {total_kv_heads=} {tp_size=}")
    if total_kv_heads >= tp_size:
        if total_kv_heads % tp_size:
            raise ValueError(
                f"{total_kv_heads} KV heads do not divide across {tp_size} ranks"
            )
        return total_kv_heads // tp_size
    if tp_size % total_kv_heads:
        raise ValueError(
            f"{tp_size} ranks do not divide across {total_kv_heads} KV heads, so "
            "the replicated case does not apply either"
        )
    return 1


@dataclass(frozen=True)
class Parallelism:
    """The widths a deployment is run at.

    Expert parallelism is a flag rather than a width because nothing here
    depends on how wide it is: it moves experts, not KV.
    """

    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    expert_parallel: bool = False

    def __post_init__(self) -> None:
        for name in ("tp_size", "pp_size", "dp_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")

    def collectives(self) -> tuple[str, ...]:
        """The collectives a step performs, named.

        The all-to-all is conditioned on the data-parallel width and not on
        expert parallelism, which is the case that reads wrong: with one
        data-parallel rank there is no all-to-all however the experts are
        arranged.

        Necessary, not sufficient, as the module docstring sets out: two
        conditions these widths cannot express also gate ATOM's peer-to-peer
        path, and the mega MoE backend takes it at one data-parallel rank
        regardless. An absence here is an absence; a presence is a candidate.
        """
        names = []
        if self.tp_size > 1:
            names.append("tp-all-reduce")
        if self.expert_parallel and self.dp_size > 1:
            names.append("moe-all-to-all")
        return tuple(names)


@dataclass(frozen=True)
class KvGeometry:
    """The paged KV one worker holds: enough to price a block, nothing more."""

    layers: int
    kv_heads: int
    head_dim: int
    element_bytes: int
    block_size: int

    def __post_init__(self) -> None:
        for name in ("layers", "kv_heads", "head_dim", "element_bytes", "block_size"):
            if getattr(self, name) < 1:
                raise ValueError(
                    f"{name} must be at least 1, got {getattr(self, name)}"
                )

    @property
    def bytes_per_token_per_layer(self) -> int:
        """Keys and values for one token in one layer."""
        return 2 * self.kv_heads * self.head_dim * self.element_bytes

    @property
    def bytes_per_block(self) -> int:
        """What one block costs: the entry size ATOM's PAGE pool is sized from."""
        return self.layers * self.block_size * self.bytes_per_token_per_layer

    @classmethod
    def from_hf_config(
        cls,
        hf_config,
        *,
        block_size: int,
        parallelism: Parallelism | None = None,
        layer_range: tuple[int, int] | None = None,
        kv_dtype=None,
    ) -> KvGeometry:
        """Read the shape off a HF config, sharded for one worker.

        `layer_range` is the half-open span of layers this pipeline stage
        holds, as ATOM's own partitioner returns it. It is required once there
        is more than one stage: defaulting to the whole stack there would size
        every stage as if it held the whole model, and the resulting block
        count would be wrong in the direction that still starts.
        """
        parallelism = Parallelism() if parallelism is None else parallelism
        text = getattr(hf_config, "text_config", hf_config)
        total = int(text.num_hidden_layers)
        if layer_range is None:
            if parallelism.pp_size > 1:
                raise ValueError(
                    f"{parallelism.pp_size} pipeline stages were declared but no "
                    "layer_range names which one this is"
                )
            layer_range = (0, total)
        start, end = layer_range
        if not 0 <= start < end <= total:
            raise ValueError(f"layer range {layer_range} is not within {total} layers")
        layers = paged_layers(text, start, end)
        head_dim = getattr(text, "head_dim", None) or (
            text.hidden_size // text.num_attention_heads
        )
        if kv_dtype is None:
            kv_dtype = getattr(text, "dtype", None)
        if kv_dtype is None:
            raise ValueError(
                "this config states no `dtype` and no kv_dtype was given, so a "
                "KV element has no size; name one of the two"
            )
        return cls(
            layers=layers,
            kv_heads=kv_heads_per_rank(
                int(text.num_key_value_heads), parallelism.tp_size
            ),
            head_dim=int(head_dim),
            element_bytes=dtype_bytes(kv_dtype),
            block_size=int(block_size),
        )
