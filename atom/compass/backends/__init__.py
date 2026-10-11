# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Cost backends: what a simulated step costs, and where that number came from.

The pieces, and the rule each exists to make structural rather than customary:

- `CostBackend` -- `estimate(batch_view) -> StepCost` plus `describe()`. Takes a
  projection of the batch, so nothing here imports the engine.
- `StepCost` / `CostTerm` -- a total folded from its parts on every read, with
  the parts re-checked on every read, so it cannot drift from them; a breakdown
  that cannot be empty; and no subclass that can shadow either.
- `Provenance` / `Species` / `Refusal` -- every number says how it was obtained
  and what produced it, with no default that would let one avoid saying.
- `Resolver` / `CostSource` -- sources consulted in a fixed order, with every
  refusal on the way down recorded on the answer rather than inferred from it.
- `KvGeometry` / `Parallelism` -- what a KV block costs and how each parallel
  width divides it, so ATOM's own pool sizing and block manager run for real
  against a stand-in model.
- `BatchView` / `RequestShape` -- the projection a backend is priced from, one
  row per request, with the batch-level sums derived from those rows and no way
  to hand them in instead. It lives here so the no-engine-imports scan, which
  reads this package and nothing else, covers it.
- `ShapeStubBackend` / `Coefficients` -- a stand-in that multiplies declared
  coefficients into a step's shapes, so a longer chunk costs longer and the
  scheduler can be caught reacting to the price. Every number it emits says it
  was declared rather than obtained, in the record and not only in the source.
- `CoarseLaw` / `CoarseBackend` -- tier a: a fitted prefill vector and a decode
  vector per capture rung, each with the convex hull it was fitted on; a step
  outside its hull is priced and its terms say `extrapolated`.
- `ProvenanceMix` -- the run-level mixture, and refusals counted by number, by
  fraction of steps and by fraction of predicted seconds. A run with nothing in
  it has no fractions and says so, rather than reporting a reassuring zero.
- `FakeModel` / `SyntheticStack` / `HfConfig` -- the stand-in itself: the
  widths are declared once and reach the geometry and the price together, so
  a pool sized for four ranks cannot be priced as one rank's. Its config is a
  published one read as JSON, or a dialled stack for shapes no released model
  has, and both are read through the same attributes.
"""

from atom.compass.backends.base import CostBackend, Tier
from atom.compass.backends.coarse import CoarseBackend, CoarseLaw
from atom.compass.backends.cost import (
    CostTerm,
    ProvenanceMix,
    StepCost,
    fold_seconds,
    fold_step,
)
from atom.compass.backends.geometry import KvGeometry, Parallelism
from atom.compass.backends.ladder import (
    CostRefused,
    CostSource,
    Resolution,
    Resolver,
)
from atom.compass.backends.model import (
    FakeModel,
    HfConfig,
    SyntheticStack,
    hf_config,
)
from atom.compass.backends.provenance import Provenance, Refusal, Species
from atom.compass.backends.shape import (
    BatchView,
    Coefficients,
    RequestShape,
    ShapeStubBackend,
)

__all__ = [
    "BatchView",
    "CoarseBackend",
    "CoarseLaw",
    "Coefficients",
    "CostBackend",
    "CostRefused",
    "CostSource",
    "CostTerm",
    "FakeModel",
    "HfConfig",
    "KvGeometry",
    "Parallelism",
    "Provenance",
    "ProvenanceMix",
    "Refusal",
    "RequestShape",
    "Resolution",
    "Resolver",
    "ShapeStubBackend",
    "Species",
    "StepCost",
    "SyntheticStack",
    "Tier",
    "fold_seconds",
    "fold_step",
    "hf_config",
]
