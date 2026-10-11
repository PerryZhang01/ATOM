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

**Two of the five `peak_torch` terms are not obtainable here.** Weights are
exact from a meta build deduped by storage -- at every width on both models
tested -- and buffers need a recording, because the formula that matched the
0.6B was 4x wrong on the 27B. A meta build needs a module
tree, which needs the engine, which this tier does not import; and there is no
recording of a card nobody has run. So `ModelTerms` takes those two terms rather
than deriving them, with no default -- the same shape as the spec's "a runtime
constant has no default" rule, and for the same reason. `from_declared_config`
fills all three with declared formulas and labels every one of them, unless a
real run measured the activation term's bytes per token in the machine spec.
"""

from __future__ import annotations

from dataclasses import dataclass

from atom.compass.backends.geometry import dtype_bytes as element_bytes
from atom.compass.memory import graph_pool
from atom.compass.memory.terms import Basis, Reading, Term

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
    does. Each is a `Term`, so each carries its own basis and its own source and
    a caller that has a meta build can hand one in beside two that are declared.
    """

    weights: Term
    buffers: Term
    activations: Term

    @classmethod
    def from_declared_config(
        cls,
        config,
        *,
        parameter_count: int,
        tp_size: int,
        warmup_tokens: int,
        dtype_bytes: int | None = None,
        config_json: dict | None = None,
        measured_activations: tuple[str, float] | None = None,
    ) -> ModelTerms:
        """All three from geometry and declared coefficients, each labelled.

        `config_json` is the model's config.json as loaded; without it, a
        field `config`'s class has a default for is refused, since the
        default cannot be told from a stated value. An MoE model, one that
        states `moe_intermediate_size`, takes its live intermediate from its
        expert fields.

        With fake models, a declared formula suffices and must say so.
        Every term below is `Basis.DECLARED` and every one names its
        successor, because none of the three is its eventual source: weights
        are owed a meta build, buffers a recording, and
        activations a liveness walk over a traced op graph *plus* the
        per-leaf invisible-scratch constants -- and the second is the one
        with no law behind it. The measured spread that makes it load-bearing
        is 0.1 KB/token on the 0.6B against 39.6 KB/token on the 27B, the
        difference between -35.0% and +3.4% held out. A formula does not
        stand in for that on a real model.

        `measured_activations`, what `MachineSpec.activations_for` returns,
        replaces the activation formula with that entry's bytes per token
        times `warmup_tokens`; the scratch is inside it.
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
            if shared is None:
                if named:
                    raise MemoryRefusal(
                        "this config states no `n_shared_experts`, and the "
                        "memory model reads it",
                        "state it in the model's config.json; this config "
                        f"names a shared expert as {', '.join(named)}, which "
                        "the memory model does not read as a count",
                    )
                if not klass.model_type:
                    # A bare PretrainedConfig names no architecture.
                    _geometry(config, file, "n_shared_experts")
                # An architecture's own config class that names no shared
                # expert, in its fields or in config.json (Qwen3MoeConfig),
                # has none: 0 is its geometry.
                shared, why = 0, f" ({klass.__name__} has none)"
            shared = int(shared)
            intermediate = (routed + shared) * int(experts)
            live = (
                f"({routed} num_experts_per_tok + {shared} n_shared_experts{why}) "
                f"x {experts} moe_intermediate_size"
            )
        weights = Term(
            "weights",
            parameter_count * dtype_bytes // tp_size,
            Basis.DECLARED,
            f"{parameter_count} stated parameters x {dtype_bytes} B / TP{tp_size}",
            "a meta build deduped by storage replaces this and is "
            "exact at every width; this shards every parameter, where a real "
            "stack replicates its norms",
        )
        buffers = Term(
            "buffers",
            positions * rotary_dim * dtype_bytes,
            Basis.DECLARED,
            f"{positions} positions x {rotary} x {dtype_bytes} B",
            "a recording off a card replaces this; this is "
            "not a recording -- it is derived from ATOM's own rotary source "
            "and validated against no card. cos and sin together are "
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
            Basis.DECLARED,
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
        if measured_activations is not None:
            entry, per_token = measured_activations
            activations = Term(
                "activations",
                int(warmup_tokens * per_token),
                Basis.SPEC,
                f"{warmup_tokens} warmup tokens x {per_token} B/token, "
                f"device.activations[{entry}].bytes_per_token[{tp_size}]",
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
    total = Reading(
        "total",
        (
            Term(
                "capacity",
                int(spec.value("device.memory.capacity_bytes")),
                Basis.SPEC,
                "device.memory.capacity_bytes",
            ),
        ),
    )
    non_torch = Reading(
        "non_torch",
        (
            Term(
                "driver and collective reserve",
                int(
                    spec.runtime_constant(
                        "driver_and_collective_reserve_bytes", tp_width
                    )
                ),
                Basis.SPEC,
                "device.runtime_constants."
                f"driver_and_collective_reserve_bytes[{tp_width}]",
            ),
        ),
    )
    peak_torch = Reading(
        "peak_torch",
        (
            model.weights,
            model.buffers,
            Term(
                "load residue",
                int(
                    spec.runtime_constant(
                        "allocator_retained_after_load_bytes", tp_width
                    )
                ),
                Basis.SPEC,
                "device.runtime_constants."
                f"allocator_retained_after_load_bytes[{tp_width}]",
            ),
            Term(
                "persistent",
                int(spec.runtime_constant("persistent_forward_buffer_bytes")),
                Basis.SPEC,
                "device.runtime_constants.persistent_forward_buffer_bytes",
            ),
            model.activations,
        ),
    )
    box = total.total - peak_torch.total - non_torch.total
    if box < 0:
        # A refusal carries its decomposition as an answer does: the reader
        # has to see which of the six terms is the one that does not fit, and
        # three of them are declared coefficients. Both readings render
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
            Term("total", total.total, Basis.DERIVED, "the total reading"),
            Term(
                "less peak_torch",
                -peak_torch.total,
                Basis.DERIVED,
                "the peak_torch reading",
            ),
            Term(
                "less non_torch",
                -non_torch.total,
                Basis.DERIVED,
                "the non_torch reading",
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
