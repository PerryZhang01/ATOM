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
| `non_torch` | the `memory_readings` entry's `driver_and_collective_reserve_bytes` |
| `cudagraph_overhead` | `graph_pool.reserves()` -- the number that reserves |
| `free` | derived as a clean box: `total - peak_torch - non_torch` |

**`free` is derived and that is load-bearing, twice over.** ATOM reads it as
`(total - free) - reserved`, and `total - free` counts every process on the
card: six prior runs died at start-up with `non_torch=152.01GB` because a
neighbour held 152 GB while this rank had reserved 2.9 GB. Taking `non_torch`
from an entry measured on a card checked quiet before and after keeps that
contamination out of the run. And with `free` derived
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
this package does not import; the simulated runner builds one and hands it in,
and `from_config` reads the weights off that tree's parameter storages. Each term
carries its own method: the buffers are analytical unless the rotary fraction
is assumed, and the activations are a declared formula unless the run's
`memory_readings` entry measured them. The capacity takes its method from the
origin the spec records for it; an entry's readings are measured, since the
entry is keyed by the configuration the run is.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from atom.compass.backends.geometry import dtype_bytes as element_bytes
from atom.compass.backends.provenance import Method
from atom.compass.memory import graph_pool
from atom.compass.memory.terms import Reading, Term, spec_term

#: What one rank's member of a `memory_readings` entry holds, in bytes but for
#: `warmup_tokens` and `record`: an engine start's readings (`compass measure
#: memory`), with the activation coefficient as `activation_bytes` over
#: `warmup_tokens`, `cudagraph_pool_bytes` null for an engine that captures
#: nothing, and `record` the digest of the start's log, as
#: `python -m atom.compass.memory.check` names a run.
ENTRY_FIELDS = (
    "capacity_bytes",
    "driver_and_collective_reserve_bytes",
    "allocator_retained_after_load_bytes",
    "persistent_forward_buffer_bytes",
    "activation_bytes",
    "warmup_tokens",
    "cudagraph_pool_bytes",
    "record",
)


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
        activations: Term,
        dtype_bytes: int | None = None,
        config_json: dict | None = None,
        tree,
    ) -> ModelTerms:
        """All three, each labelled with where its bytes came from.

        `config_json` is the model's config.json as loaded; without it, a
        field `config`'s class has a default for is refused, since the
        default cannot be told from a stated value.

        `tree` is the model's module tree built on the meta device, with
        ATOM's own quantisation config and process groups; the weights are the
        bytes of its parameter storages, each storage once, so a tied
        `lm_head` that shares the embedding's storage counts once, as it is
        resident once.

        `activations` is what `MemoryReadings.activations` gives: a real run's
        warmup peak, the invisible scratch inside it. No form in the model's
        geometry stands in for it; 0.1 KB/token of scratch on the 0.6B
        against 39.6 KB/token on the 27B is the difference between -35.0% and
        +3.4% held out. Assumed buffers are owed a config that states the
        rotary fraction.
        """
        file = config_json or {}
        file = file.get("text_config", file)
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
        return cls(weights, buffers, activations)


@dataclass(frozen=True, slots=True)
class MemoryReadings:
    """One rank's member of a `memory_readings` entry, and the entry it is from.

    Each reading is a term of the run's readings, measured: the entry is keyed
    by the model, platform, layout and engine config the run is.
    """

    values: Mapping[str, object]
    source: str

    @classmethod
    def from_member(cls, member: object, source: str) -> MemoryReadings:
        """The readings of a member, or a refusal naming what it holds instead."""
        fields = set(member) if isinstance(member, Mapping) else None
        if fields != set(ENTRY_FIELDS):
            raise MemoryRefusal(
                f"{source} holds {sorted(fields or ())}, not one rank's readings",
                f"a member states exactly {', '.join(ENTRY_FIELDS)}; take the "
                "entry again with compass measure memory",
            )
        for name in ENTRY_FIELDS:
            value = member[name]
            whole = isinstance(value, int) and not isinstance(value, bool)
            if name == "record":
                whole = isinstance(value, str) and value.startswith("sha256:")
            elif whole:
                whole = value >= (name == "warmup_tokens")
            if not (whole or (value is None and name == "cudagraph_pool_bytes")):
                raise MemoryRefusal(
                    f"{source} holds {name} {value!r}, which is no reading",
                    "a reading is a whole number of bytes, or of warmup tokens, "
                    "and the record a log's sha256",
                )
        return cls(dict(member), source)

    def term(self, name: str, field: str) -> Term:
        """The reading `field` as the term `name`."""
        return Term(
            name,
            self.values[field],
            Method.MEASURED,
            f"{self.source} {field}",
            records=(self.values["record"],),
        )

    def activations(self, warmup_tokens: int) -> Term:
        """The activations at `warmup_tokens`: the reading at its own warmup
        tokens, or the coefficient times these, an estimate."""
        nbytes, tokens = self.values["activation_bytes"], self.values["warmup_tokens"]
        source = f"{self.source} activation_bytes over {tokens} warmup tokens"
        records = (self.values["record"],)
        if warmup_tokens == tokens:
            return Term("activations", nbytes, Method.MEASURED, source, records=records)
        return Term(
            "activations",
            round(nbytes * warmup_tokens / tokens),
            Method.ESTIMATED,
            f"{source}, x {warmup_tokens} warmup tokens",
            records=records,
        )


@dataclass(frozen=True, slots=True)
class DeviceReadings:
    """The five readings, each as its terms, for one width of one spec."""

    total: Reading
    peak_torch: Reading
    non_torch: Reading
    cudagraph_overhead: Reading
    free: Reading
    tp_width: int | str
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
    memory: MemoryReadings,
    tp_width: int | str,
    model: ModelTerms,
    cudagraph_overhead: Reading,
) -> DeviceReadings:
    """The five readings: the capacity off `spec`, the reserve, residue and
    persistent buffers off `memory`, the run's `memory_readings` member.

    `cudagraph_overhead` is the reading `graph_pool.reserves()` returns, and a
    reading from `graph_pool.predicts()` is refused here by name. The two
    disagree by 4-19x and only the first is what ATOM subtracts from the budget,
    so which one reserves is settled at this call site by the code rather than
    by a comment beside it.

    `tp_width` labels the table: the width, or under DP-attention the layout,
    `"<tp>x<dp>"`, or `"<tp>x<dp>x<ep>"` with expert parallelism.
    """
    if isinstance(tp_width, int) and tp_width < 1:
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
    capacity = "device.memory.capacity_bytes"
    total = Reading(
        "total",
        (spec_term(spec, "capacity", capacity, int(spec.value(capacity))),),
    )
    non_torch = Reading(
        "non_torch",
        (
            memory.term(
                "driver and collective reserve", "driver_and_collective_reserve_bytes"
            ),
        ),
    )
    peak_torch = Reading(
        "peak_torch",
        (
            model.weights,
            model.buffers,
            memory.term("load residue", "allocator_retained_after_load_bytes"),
            memory.term("persistent", "persistent_forward_buffer_bytes"),
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
