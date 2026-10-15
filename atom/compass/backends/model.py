# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The stand-in model itself: one declaration, the geometry and the price.

The two halves a simulated run needs already exist in this package and are
built separately. `KvGeometry.from_hf_config` says what a KV block costs at a
set of parallel widths; `StepStubBackend` says what a step costs, and is told
the widths a second time so it knows which collectives to charge. Nothing
joined them, and the join is not bookkeeping:

    geometry = KvGeometry.from_hf_config(config, block_size=64,
                                         parallelism=Parallelism(tp_size=4))
    backend = StepStubBackend(geometry=geometry)

is accepted today. The pool is sized for four ranks and the price is a single
rank's, with no all-reduce charged at all, and nothing anywhere can notice: a
`KvGeometry` is five integers and does not record the widths it was divided
by. `FakeModel` takes the widths once and hands the same `Parallelism` object
to both, so the pair cannot be built disagreeing.

**A stand-in, not a model.** It reads no checkpoint, holds no weights and
allocates nothing. What it is for is that everything downstream of it is then
real -- ATOM's own pool sizing, block manager and scheduler run against a
genuine number per width rather than one this package decided -- and what it
says about a duration comes from declared coefficients and is not a prediction
of anything. Both halves say so themselves; `describe()` repeats it in the one
line a run record keeps.

Two config sources, because the milestone needs both:

- **A published config**, read as JSON through `hf_config`. The geometry is
  read off the real layer, head and dtype numbers, so a later backend is a
  swap rather than a rebuild. The reader is twenty lines of attribute access
  rather than a modelling library, which keeps this package importable on a
  machine that has neither the engine nor an accelerator.
- **`SyntheticStack`**, dialled rather than published, for shapes no released
  model has: a hundred-layer stack on one KV head, a head dimension nothing
  ships, a hybrid whose interval is whatever a test needs. It exposes the
  attribute names a config uses and is read by the same code path, so it
  exercises the reader rather than bypassing it.

Two layer counts, and each half reads its own. `geometry.layers` counts the
layers holding a cache of every past token, which is what a KV block is sized
from; `stage_layers` counts the layers this worker runs, which is what a
collective is charged on, since an all-reduce runs on a layer keeping a bounded
recurrent state exactly as it does on a paged one. On a stack that is one
full-attention layer in four the two differ by four and on a uniform stack they
coincide, so neither is recoverable from the other and both are handed on: the
geometry to the pool, the span to the price. Both stay readable here rather
than one being implied by the other.

Nothing constrains where the config came from, either. The `config` argument
is unannotated and read by attribute name, so a config type defined beside the
engine -- importing the engine, and buildable only on a machine that has one
-- reaches this constructor and is accepted. Neither guard on this package can
see that: the import scan reads the imports in these files and there are none
to read, and the test that checks where the types crossing this boundary are
defined reads definition sites of types this package names, not the type of
whatever an argument was handed. Closing it needs an instrument that reads
what is passed rather than what is imported. There is none here, and the route
is open.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

from atom.compass.backends.geometry import (
    _PAGED_LAYER_KINDS,
    _UNCACHED_LAYER_KINDS,
    KvGeometry,
    Parallelism,
)
from atom.compass.backends.step_stub import Coefficients, StepStubBackend

# Layer kinds a dialled stack is built out of, read from the module that
# decides what they mean rather than spelled again here. Which strings name a
# layer holding a cache of every past token, and which name one holding none,
# is the reader's to say; a second spelling beside this dial would be tied to
# the first by nothing, and the first divergence would be this dial emitting a
# kind its own reader refuses -- the one outcome the dial is meant not to be
# able to produce. Either set may carry several spellings of the same kind and
# the reader treats them alike, so the dial takes the first of each in sorted
# order and any of them would do. The interval names how often the paged one
# appears.
_PAGED = min(_PAGED_LAYER_KINDS)
_UNCACHED = min(_UNCACHED_LAYER_KINDS)


class HfConfig:
    """Attribute access over a decoded config, and nothing else.

    A published config is a JSON object of plain values and nested objects,
    and the geometry reads it by attribute name. That is the whole interface,
    so it is met by twenty lines here rather than by a modelling library: this
    package is meant to import on a machine with no engine and no accelerator,
    and a dependency that pulls in a framework to read five integers works
    against that.

    A key that is absent raises `AttributeError`, which is what makes the
    geometry's own optional reads work -- `getattr(config, "layer_types",
    None)` has to come back `None` on a uniform stack rather than raise, and
    `head_dim` has to be absent-able so the fallback through `hidden_size`
    runs. The message names the key and the keys that are present, because the
    usual cause is a config nested one level further down than expected.
    """

    def __init__(self, entries: Mapping[str, object]) -> None:
        if not isinstance(entries, Mapping):
            raise TypeError(f"a config is a mapping of names, not {entries!r}")
        self._entries = dict(entries)

    def __getattr__(self, name: str) -> object:
        entries = self.__dict__.get("_entries", {})
        if name not in entries:
            raise AttributeError(
                f"this config has no {name}; it carries "
                f"{', '.join(sorted(entries)) or 'nothing'}"
            )
        value = entries[name]
        return HfConfig(value) if isinstance(value, Mapping) else value

    def __repr__(self) -> str:
        return f"HfConfig({', '.join(sorted(self._entries))})"


def hf_config(source) -> HfConfig:
    """A published config, from a path to its JSON or from the decoded object."""
    if isinstance(source, Mapping):
        return HfConfig(source)
    return HfConfig(json.loads(Path(source).read_text()))


@dataclass(frozen=True)
class SyntheticStack:
    """A dialled stand-in config, read by the code that reads a published one.

    Every field is spelled the way a config spells it, so the geometry reads
    this through the same attributes and the same fallbacks; there is no
    branch anywhere for a synthetic config, and a dial that produced a shape
    the reader refuses is refused by the reader.

    `full_attention_interval` is the hybrid dial: `None` is a uniform stack
    that names no layer kinds, and an interval of `n` makes every `n`-th layer
    the one that holds a cache of every past token, which is the pattern a
    published hybrid uses. Setting it fills `layer_types` in, so the stack
    that names an interval and no kinds -- the one the geometry refuses
    because which of its layers are paged is then unstated -- is not
    reachable from here.

    `head_dim` unset is the other published shape: a config may state the head
    dimension or leave it out, and left out the reader divides `hidden_size`
    by the query head count. That division is the only fallback the reader
    has, so leaving `head_dim` unset is the only position from which
    `hidden_size` changes a number -- which is why it is the default. State
    `head_dim` and `hidden_size` is carried and read by nobody, exactly as it
    is in a published config that states both.
    """

    num_hidden_layers: int = 32
    num_key_value_heads: int = 8
    num_attention_heads: int = 32
    head_dim: int | None = None
    hidden_size: int = 4096
    dtype: str = "bfloat16"
    full_attention_interval: int | None = None

    def __post_init__(self) -> None:
        for spec in fields(self):
            if spec.name in ("dtype", "full_attention_interval"):
                continue
            value = getattr(self, spec.name)
            if spec.name == "head_dim" and value is None:
                continue
            if not isinstance(value, int):
                raise TypeError(
                    f"{spec.name} must be a whole number of at least 1, not {value!r}"
                )
            if value < 1:
                raise ValueError(f"{spec.name} must be at least 1, got {value}")
        if self.head_dim is None and self.hidden_size < self.num_attention_heads:
            raise ValueError(
                f"head_dim is unset, so it is {self.hidden_size} hidden units "
                f"divided by {self.num_attention_heads} query heads, which is "
                "less than one unit a head; widen hidden_size or state head_dim"
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                f"{self.num_attention_heads} query heads do not group over "
                f"{self.num_key_value_heads} KV heads; grouped-query attention "
                "is what bounds the tensor-parallel split and a stack that does "
                "not group has no such bound to dial"
            )
        if self.full_attention_interval is None:
            return
        if not 1 <= self.full_attention_interval <= self.num_hidden_layers:
            raise ValueError(
                f"a full-attention layer every {self.full_attention_interval} "
                f"of {self.num_hidden_layers} layers places none at all; a stack "
                "with no paged layer holds no KV and sizes no pool"
            )

    @property
    def layer_types(self) -> list[str] | None:
        """The per-layer kinds, or `None` for a stack that is uniform.

        `None` rather than a list of one repeated kind, because that is what a
        uniform published config carries and the point of this type is to be
        read by the same code.
        """
        interval = self.full_attention_interval
        if interval is None:
            return None
        return [
            _PAGED if (index + 1) % interval == 0 else _UNCACHED
            for index in range(self.num_hidden_layers)
        ]


class FakeModel:
    """The stand-in a simulated run is built from: geometry and price together.

    One `Parallelism` goes in and the same object reaches both halves, so the
    pool this sizes and the collectives this charges are widths of one
    deployment rather than two that happen to have been typed alike.

    `layer_range` is the half-open span of layers this worker holds, as ATOM's
    own partitioner returns it, and it is handed in rather than derived -- the
    partitioning is the engine's and a second implementation of it here would
    be a second answer. The refusal when more than one stage is declared and
    no range names which one is the geometry's, reached through this
    constructor unchanged.
    """

    def __init__(
        self,
        config,
        *,
        block_size: int,
        parallelism: Parallelism | None = None,
        coefficients: Coefficients | None = None,
        layer_range: tuple[int, int] | None = None,
        kv_dtype=None,
    ) -> None:
        self.config = config
        self.block_size = int(block_size)
        self.parallelism = Parallelism() if parallelism is None else parallelism
        self.coefficients = Coefficients() if coefficients is None else coefficients
        self.geometry = KvGeometry.from_hf_config(
            config,
            block_size=self.block_size,
            parallelism=self.parallelism,
            layer_range=layer_range,
            kv_dtype=kv_dtype,
        )
        text = getattr(config, "text_config", config)
        # The whole stack the config declares, which is what a set of pipeline
        # spans has to add up to; `stage_layers` below is this worker's share.
        self.stack_depth = int(text.num_hidden_layers)
        self.layer_range = (0, self.stack_depth) if layer_range is None else layer_range
        self.kv_dtype = kv_dtype
        # The span reaches the price and the geometry reaches the pool. The
        # backend is built after both are known because the span is what it
        # charges a collective on, and this worker's span is the one it runs.
        self.backend = StepStubBackend(
            coefficients=self.coefficients,
            parallelism=self.parallelism,
            geometry=self.geometry,
            stack_layers=self.stage_layers,
        )

    @classmethod
    def from_json(cls, source, **declared) -> FakeModel:
        """This model over a published config, read from its JSON."""
        return cls(hf_config(source), **declared)

    @property
    def stage_layers(self) -> int:
        """Layers this worker holds, of every kind.

        `geometry.layers` is the subset of these that holds a cache of every
        past token, and on a hybrid the two are different numbers: a layer
        that keeps a bounded recurrent state still runs, still has weights and
        still takes part in a collective, it just costs a block nothing. This
        count is what the price charges a collective on, and the subset is
        what the pool is sized from.
        """
        start, end = self.layer_range
        return end - start

    def stages(self, layer_ranges: Sequence[tuple[int, int]]) -> tuple[FakeModel, ...]:
        """One model per pipeline stage, over the spans the partitioner gave.

        The spans are checked twice, because they fail in two ways and only
        one of them is a count. A four-stage deployment handed three spans
        builds three workers that each size a pool and a fourth that is never
        built; the count check catches that. The likelier mistake is the right
        number of well-formed spans that tile the wrong stack -- the spans a
        partitioner returns when it is handed the wrong depth. Each of those
        satisfies the geometry's own `0 <= start < end <= total` on its own,
        so nothing per-span refuses them, and the pools they size then hold a
        fraction of the model's KV with nothing anywhere saying so. So the
        spans are also checked as a partition: sorted, they start at 0, each
        begins where the last ended, and the last ends at the declared depth.

        They are still handed in rather than derived. The partitioning is the
        engine's answer and a second implementation of it here would be a
        second answer; checking an answer is not computing one.
        """
        ranges = tuple(layer_ranges)
        if len(ranges) != self.parallelism.pp_size:
            raise ValueError(
                f"{len(ranges)} layer ranges for {self.parallelism.pp_size} "
                "pipeline stages; every stage holds a span and the partitioner "
                "names them all"
            )
        ordered = sorted(ranges)
        boundaries = [0] + [end for _, end in ordered]
        if [start for start, _ in ordered] != boundaries[:-1] or (
            boundaries[-1] != self.stack_depth
        ):
            raise ValueError(
                f"layer ranges {list(ranges)} do not partition the "
                f"{self.stack_depth}-layer stack; every layer belongs to one "
                "stage, so sorted the spans start at 0, each begins where the "
                "last ended, and the last ends at the full depth"
            )
        return tuple(
            FakeModel(
                self.config,
                block_size=self.block_size,
                parallelism=self.parallelism,
                coefficients=self.coefficients,
                layer_range=span,
                kv_dtype=self.kv_dtype,
            )
            for span in ranges
        )

    def describe(self) -> str:
        """One line for the run record: what it holds, and what it is not."""
        start, end = self.layer_range
        held = (
            f"layers {start}-{end} ({self.geometry.layers} paged"
            f" of {self.stage_layers})"
        )
        heads = "head" if self.geometry.kv_heads == 1 else "heads"
        block = (
            f"{self.geometry.kv_heads} KV {heads} x {self.geometry.head_dim}"
            f" at {self.geometry.element_bytes} B,"
            f" {self.geometry.block_size}-token blocks"
            f" = {self.geometry.bytes_per_block} B/block"
        )
        return f"stand-in model, {held}, {block} | {self.backend.describe()}"
