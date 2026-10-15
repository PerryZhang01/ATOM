# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The five readings `get_num_blocks` takes off a card, taken off a spec instead.

The rule is **substitute the readings, never the arithmetic**. ATOM's
`ModelRunner.get_num_blocks` (`atom/model_engine/model_runner.py`) is five
device readings and then some arithmetic over them; this module owns the five,
and the budget formula, the 2% margin, the `min(budget, free)` clamp and
`plan_pools` stay ATOM's. A copy of that formula here is exactly the drift the
substitution exists to avoid, and upstream will change it.

| Reading | Where it comes from now |
|---|---|
| `total` | `device.memory.capacity_bytes` |
| `peak_torch` | weights + buffers + load residue + persistent + activations |
| `non_torch` | `driver_and_collective_reserve_bytes`, a width table |
| `cudagraph_overhead` | `graph_pool.reserves()` -- the number that reserves |
| `free` | derived as a clean box: `total - peak_torch - non_torch` |

**`free` is derived and that is load-bearing, twice over.** ATOM reads it as
`(total - free) - reserved`, and `total - free` counts every process on the
card: six prior runs died at start-up with `non_torch=152.01GB` because a
neighbour held 152 GB while this rank had reserved 2.9 GB. Taking `non_torch`
from a spec removes that contamination by construction. And with `free` derived
as the box that is left, the `min(budget, free)` clamp cannot bind -- the prior
design needed a `free_was_binding()` guard to refuse records where it had, and
this one does not. Making it inert is this module's job; ATOM's own budget
arithmetic runs over these readings in `atom.compass.runner.overrides`, and
`test_the_min_budget_free_clamp_does_not_bind_on_either_side_of_free` in
`tests/compass/test_kv_budget_engine.py` checks that the clamp does not bind.

The declared scope boundary that follows, stated so it is not discovered as a
gap: Compass models a **dedicated** device. It will not predict the OOM a shared
box produces and will not reproduce a neighbour-induced admission cliff.

**The weights are not derivable here.** They are exact from a meta build
deduped by storage, which needs a module tree, which needs the engine, which
this package does not import; the simulated runner builds one and hands it in.
`from_config` reads the weights off that tree's parameter storages when it is
given one, and otherwise declares them from a stated parameter count. Each term
carries its own method: the buffers are analytical unless the rotary fraction
is assumed, and the activations are a declared formula unless a real run
measured their bytes per token in the machine spec. The spec fields used as
terms take their method from the origin the spec records for them.
"""

from __future__ import annotations

from dataclasses import dataclass

from atom.compass.backends.geometry import dtype_bytes as element_bytes
from atom.compass.backends.provenance import Method
from atom.compass.memory import graph_pool
from atom.compass.memory.terms import Reading, Term, spec_term

#: Gate and up are both live before their product, so the MLP's intermediate
#: activation is resident twice at the peak. Geometry, not a coefficient.
_LIVE_INTERMEDIATE = 2


class MemoryRefusal(Exception):
    """A declined reading that names what it could not do and what would fix it."""

    def __init__(self, what: str, remedy: str) -> None:
        self.what = what
        self.remedy = remedy
        super().__init__(f"{what}. {remedy}")


def _stated(config, file, name: str):
    """One field off a HF config, or None when the model does not state it.

    A model's config class fills in its own default for a field the
    checkpoint's config.json leaves out, and the object cannot tell the two
    apart, so a field the class has a default for counts only when `file`,
    the keys of that config.json, names it. DeepSeek-V4 loads through
    `DeepseekV3Config`, whose `intermediate_size` default is not V4's geometry.
    """
    text = getattr(config, "text_config", config)
    value = getattr(text, name, None)
    if value is not None and hasattr(type(text), name) and name not in file:
        return None
    return value


def _geometry(config, file, name: str):
    """One stated field off a HF config, refusing by name rather than defaulting."""
    value = _stated(config, file, name)
    if value is None:
        raise MemoryRefusal(
            f"this config states no `{name}`, and the memory model reads it",
            "state it in the model's config.json, and build the config through "
            "ATOM's own config classes with that file beside it; a config class "
            "default is not the model's geometry",
        )
    return value


def _dtype(config):
    """The dtype a model's tensors are resident at."""
    text = getattr(config, "text_config", config)
    value = getattr(text, "dtype", None)
    if value is None:
        raise MemoryRefusal(
            "this config states no `dtype`, and every byte of the model-side "
            "terms is twice or half what it should be without it",
            "name the dtype on the config, or pass `dtype_bytes` explicitly",
        )
    return value


@dataclass(frozen=True, slots=True)
class ModelTerms:
    """The three `peak_torch` terms that belong to the model, not to the card.

    Kept together because they move together -- they are the same bytes on any
    card, and only the other two terms of `peak_torch` change when the spec
    does. Each is a `Term`, so each carries its own method and its own source.
    """

    weights: Term
    buffers: Term
    activations: Term

    @classmethod
    def from_config(
        cls,
        config,
        *,
        parameter_count: int | None = None,
        tp_size: int,
        warmup_tokens: int,
        dtype_bytes: int | None = None,
        config_json: dict | None = None,
        activations_entry: tuple[str, float, tuple[str, ...]] | None = None,
        tree=None,
    ) -> ModelTerms:
        """All three, each labelled with where its bytes came from.

        `config_json` is the model's config.json as loaded; without it, a
        field `config`'s class has a default for is refused, since the
        default cannot be told from a stated value. An MoE model, one that
        states `moe_intermediate_size`, takes its live intermediate from its
        expert fields, or from `intermediate_size` when it states dense layers
        wider than that.

        `tree` is the model's module tree built on the meta device, with
        ATOM's own quantisation config and process groups; the weights are the
        bytes of its parameter storages, each storage once, so a tied
        `lm_head` that shares the embedding's storage counts once, as it is
        resident once. Without it, the weights are `parameter_count` at one
        dtype, sharded by `tp_size`.

        Every declared term below names its successor: stated weights are
        owed a meta build, assumed buffers a config that states the rotary
        fraction, and the activation formula an entry measured on the model
        or a liveness walk over a traced op graph *plus* the per-leaf
        invisible-scratch constants -- and the second is the one with no law
        behind it. The measured spread that makes it load-bearing is
        0.1 KB/token on the 0.6B against 39.6 KB/token on the 27B, the
        difference between -35.0% and +3.4% held out. A formula does not
        stand in for that on a real model.

        `activations_entry`, what `MachineSpec.activations_for` returns,
        replaces the activation formula with that entry's bytes per token
        times `warmup_tokens`, an estimate; the scratch is inside it. The term
        keeps the records of the runs the entry was read off.
        """
        if tp_size < 1:
            raise ValueError(f"tensor-parallel width is at least 1: {tp_size}")
        file = config_json or {}
        file = file.get("text_config", file)
        hidden = int(_geometry(config, file, "hidden_size"))
        positions = int(_geometry(config, file, "max_position_embeddings"))
        if dtype_bytes is None:
            dtype_bytes = element_bytes(_dtype(config))
        rope = _stated(config, file, "qk_rope_head_dim")
        assumed = ""
        if rope is not None:
            # An MLA model rotates only this slice of each head.
            rotary_dim = int(rope)
            rotary = f"{rotary_dim} qk_rope_head_dim"
        else:
            head_dim = int(_geometry(config, file, "head_dim"))
            # 1.0 is the right reading for a model with full rotary, so an
            # absent field is not refused here. But a config that states 1.0
            # and one that states nothing must not render the same row: the
            # second is the absence the recorded 4x came from, and an
            # assumption that does not appear in the table is not an
            # assumption a reader can see.
            stated = _stated(config, file, "partial_rotary_factor")
            partial = 1.0 if stated is None else float(stated)
            assumed = "" if stated is not None else " (absent from config, assumed)"
            rotary_dim = int(head_dim * partial)
            rotary = (
                f"int({head_dim} head_dim x {partial} "
                f"partial_rotary_factor{assumed})"
            )
        experts = _stated(config, file, "moe_intermediate_size")
        if experts is None:
            intermediate = int(_geometry(config, file, "intermediate_size"))
            live = f"{intermediate} intermediate"
        else:
            # Each token's routed experts and the shared ones are live at once.
            routed = int(_geometry(config, file, "num_experts_per_tok"))
            text = getattr(config, "text_config", config)
            klass = type(text)
            named = sorted(k for k in vars(text) if "shared" in k)
            shared, why = _stated(config, file, "n_shared_experts"), ""
            if shared is None and hasattr(klass, "shared_expert_intermediate_size"):
                # One shared expert sized by its own width (Qwen2MoeConfig and
                # its successors), live beside the routed ones.
                width = int(_geometry(config, file, "shared_expert_intermediate_size"))
                intermediate = routed * int(experts) + width
                live = (
                    f"({routed} num_experts_per_tok x {experts} moe_intermediate_size"
                    f" + {width} shared_expert_intermediate_size)"
                )
            elif shared is None:
                if named:
                    raise MemoryRefusal(
                        "this config states no `n_shared_experts`, and the "
                        "memory model reads it",
                        "state it in the model's config.json; this config "
                        f"names a shared expert as {', '.join(named)}",
                    )
                if not klass.model_type:
                    # A bare PretrainedConfig names no architecture.
                    _geometry(config, file, "n_shared_experts")
                # An architecture's own config class that names no shared
                # expert, in its fields or in config.json (Qwen3MoeConfig),
                # has none: 0 is its geometry.
                shared, why = 0, f" ({klass.__name__} has none)"
            if shared is not None:
                shared = int(shared)
                intermediate = (routed + shared) * int(experts)
                live = (
                    f"({routed} num_experts_per_tok + {shared} n_shared_experts{why}) "
                    f"x {experts} moe_intermediate_size"
                )
            # ATOM builds a dense MLP in these layers instead; the wider form
            # is the one live at the peak.
            if (
                _stated(config, file, "first_k_dense_replace")
                or _stated(config, file, "mlp_only_layers")
                or (_stated(config, file, "decoder_sparse_step") or 1) > 1
            ):
                dense = int(_geometry(config, file, "intermediate_size"))
                intermediate = max(dense, intermediate)
                live = f"max({dense} intermediate_size, {live})"
        if tree is not None:
            # Parameters that share a storage (a tied head, a view) are one
            # allocation; `_cdata` names the storage even on meta, where every
            # data pointer is zero.
            storages = {
                p.untyped_storage()._cdata: p.untyped_storage().nbytes()
                for p in tree.parameters()
            }
            weights = Term(
                "weights",
                sum(storages.values()),
                Method.ANALYTICAL,
                f"{len(storages)} parameter storages of a meta build of "
                f"{type(tree).__name__}",
            )
        else:
            weights = Term(
                "weights",
                parameter_count * dtype_bytes // tp_size,
                Method.DECLARED,
                f"{parameter_count} stated parameters x {dtype_bytes} B / "
                f"TP{tp_size}",
                "the run file states the parameter count; "
                "a meta build deduped by storage replaces this and is "
                "exact at every width; this shards every parameter, where a "
                "real stack replicates its norms",
            )
        buffers = Term(
            "buffers",
            positions * rotary_dim * dtype_bytes,
            Method.DECLARED if assumed else Method.ANALYTICAL,
            f"{positions} positions x {rotary} x {dtype_bytes} B",
            (
                "partial_rotary_factor 1.0 is assumed; a config that states it "
                "replaces this. "
                if assumed
                else ""
            )
            + "cos and sin together are "
            "positions x rotary_dim elements, because inv_freq holds "
            "rotary_dim/2 of them (RotaryEmbedding._compute_inv_freq in "
            "atom/model_ops/rotary_embedding.py), and "
            "they are resident at the model dtype they are cast to, not the "
            "fp32 they are computed in (RotaryEmbedding.__init__, with the "
            "default dtype set in ModelRunner.__init__)",
        )
        activations = Term(
            "activations",
            int(
                warmup_tokens
                * dtype_bytes
                * (
                    graph_pool.LIVE_TENSORS_PER_LAYER * hidden
                    + _LIVE_INTERMEDIATE * intermediate
                )
            ),
            Method.DECLARED,
            f"{warmup_tokens} warmup tokens x {dtype_bytes} B x "
            f"({graph_pool.LIVE_TENSORS_PER_LAYER} x {hidden} hidden + "
            f"{_LIVE_INTERMEDIATE} x {live})",
            "a device.activations entry, or a liveness walk over a traced op "
            "graph plus the per-leaf "
            "invisible-scratch constants, replace this; the per-layer "
            "coefficient is ATOM's own "
            "(ModelRunner._piecewise_per_token_bytes), over one live "
            "layer rather than all of them",
        )
        if activations_entry is not None:
            entry, per_token, records = activations_entry
            activations = Term(
                "activations",
                int(warmup_tokens * per_token),
                Method.ESTIMATED,
                f"{warmup_tokens} warmup tokens x {per_token} B/token, "
                f"device.activations[{entry}].bytes_per_token[{tp_size}]",
                records=records,
            )
        return cls(weights, buffers, activations)


@dataclass(frozen=True, slots=True)
class DeviceReadings:
    """The five readings, each as its terms, for one width of one spec."""

    total: Reading
    peak_torch: Reading
    non_torch: Reading
    cudagraph_overhead: Reading
    free: Reading
    tp_width: int
    spec_digest: str

    def as_dict(self) -> dict[str, Reading]:
        """The five by the name `get_num_blocks` knows each of them by."""
        return {
            "total": self.total,
            "peak_torch": self.peak_torch,
            "non_torch": self.non_torch,
            "cudagraph_overhead": self.cudagraph_overhead,
            "free": self.free,
        }

    @property
    def declared(self) -> tuple[str, ...]:
        """Every term across the five still standing on a declared coefficient."""
        return tuple(
            f"{name}.{term}"
            for name, reading in self.as_dict().items()
            for term in reading.declared
        )

    def table(self) -> str:
        """Every reading as its terms. There is no rendering that is five totals."""
        head = [f"device readings at TP{self.tp_width}, spec {self.spec_digest}"]
        return "\n".join(head + [r.table() for r in self.as_dict().values()])

    def __str__(self) -> str:
        return self.table()


def device_readings(
    spec,
    *,
    tp_width: int,
    model: ModelTerms,
    cudagraph_overhead: Reading,
) -> DeviceReadings:
    """The five readings for one width, or a refusal naming the field and width.

    `cudagraph_overhead` is the reading `graph_pool.reserves()` returns, and a
    reading from `graph_pool.predicts()` is refused here by name. The two
    disagree by 4-19x and only the first is what ATOM subtracts from the budget,
    so which one reserves is settled at this call site by the code rather than
    by a comment beside it.

    A width absent from either width table refuses, and the spec's own refusal
    names the field and the width -- `driver_and_collective_reserve_bytes` and
    `allocator_retained_after_load_bytes` are `Kind.WIDTH_TABLE` because no
    fixed-plus-per-peer form fits 5980/6340/9138 at widths 2/4/8, so there is
    nothing to interpolate along. Nothing here catches that refusal.
    """
    if tp_width < 1:
        raise ValueError(f"tensor-parallel width is at least 1: {tp_width}")
    if cudagraph_overhead.name != graph_pool.RESERVES:
        raise MemoryRefusal(
            f"the graph-pool reading given is {cudagraph_overhead.name!r}, and "
            f"only {graph_pool.RESERVES!r} reserves anything",
            "pass `graph_pool.reserves(...)`, which mirrors ATOM's own "
            "estimator; `graph_pool.predicts(...)` is what the pool really "
            "costs and the two disagree by 4-19x, so substituting it would "
            "produce a block count ATOM would never reach",
        )
    rc = "device.runtime_constants."
    reserve = "driver_and_collective_reserve_bytes"
    residue = "allocator_retained_after_load_bytes"
    persistent = "persistent_forward_buffer_bytes"
    capacity = "device.memory.capacity_bytes"
    total = Reading(
        "total",
        (spec_term(spec, "capacity", capacity, int(spec.value(capacity))),),
    )
    non_torch = Reading(
        "non_torch",
        (
            spec_term(
                spec,
                "driver and collective reserve",
                rc + reserve,
                int(spec.runtime_constant(reserve, tp_width)),
                width=tp_width,
            ),
        ),
    )
    peak_torch = Reading(
        "peak_torch",
        (
            model.weights,
            model.buffers,
            spec_term(
                spec,
                "load residue",
                rc + residue,
                int(spec.runtime_constant(residue, tp_width)),
                width=tp_width,
            ),
            spec_term(
                spec,
                "persistent",
                rc + persistent,
                int(spec.runtime_constant(persistent)),
            ),
            model.activations,
        ),
    )
    box = total.total - peak_torch.total - non_torch.total
    if box < 0:
        # A refusal carries its decomposition as an answer does: the reader
        # has to see which of the six terms is the one that does not fit, and
        # how each was obtained. Both readings render
        # themselves, so the decomposition costs a newline.
        needed = (peak_torch.total + non_torch.total) / total.total
        raise MemoryRefusal(
            f"the non-KV footprint at TP{tp_width} does not fit the card, so "
            f"the clean box is negative and there is no free memory to "
            f"report:\n{total.table()}\n{peak_torch.table()}\n"
            f"{non_torch.table()}",
            "this configuration does not start on this card. Even "
            "--gpu-memory-utilization 1.0 is insufficient -- the non-KV terms "
            f"alone are {needed:.2f} of total -- so the lever ATOM names on "
            "its own version of this failure (ModelRunner.get_num_blocks) will "
            "not reach it; reduce the width, the model or the warmup shape, "
            "or name a larger card. A clamped zero here would be a free "
            "reading nobody could read as a refusal",
        )
    free = Reading(
        "free",
        (
            total.as_term("total", source="the total reading"),
            peak_torch.as_term(
                "less peak_torch", sign=-1, source="the peak_torch reading"
            ),
            non_torch.as_term(
                "less non_torch", sign=-1, source="the non_torch reading"
            ),
        ),
    )
    return DeviceReadings(
        total=total,
        peak_torch=peak_torch,
        non_torch=non_torch,
        cudagraph_overhead=cudagraph_overhead,
        free=free,
        tp_width=tp_width,
        spec_digest=spec.digest(),
    )
