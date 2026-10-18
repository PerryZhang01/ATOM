# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Cost backends: what a simulated step costs, and where that number came from.

The pieces, and the rule each exists to make structural rather than customary:

- `CostBackend` -- `price(batch_view) -> StepCost` plus `describe()`, and a
  `Granularity`, step or op. Takes a projection of the batch, so nothing here
  imports the engine.
- `StepCost` / `CostTerm` -- a total folded from its parts on every read, with
  the parts re-checked on every read, so it cannot drift from them; a breakdown
  that cannot be empty; and no subclass that can shadow either.
- `Provenance` / `Method` / `Refusal` -- every number says how it was obtained
  (analytical, declared, measured or estimated, and whether an estimate is
  validated) and what produced it, with no default that would let one avoid
  saying.
- `Resolver` / `CostSource` -- sources consulted in a fixed order, with every
  refusal on the way down recorded on the answer rather than inferred from it.
- `KvGeometry` / `Parallelism` -- what a KV block costs and how each parallel
  width divides it, so ATOM's own pool sizing and block manager run for real
  against a stand-in model.
- `BatchView` / `RequestShape` -- the projection a backend is priced from, one
  row per request, with the batch-level sums derived from those rows and no way
  to hand them in instead. It lives here so the no-engine-imports scan, which
  reads this package and nothing else, covers it.
- `StepStubBackend` / `Coefficients` -- a stand-in that multiplies stated
  coefficients into a step's shapes, so a longer chunk costs longer and the
  scheduler can be caught reacting to the price. Every number it emits is
  `declared`, in the record and not only in the source.
- `StepLinearLaw` / `StepLinearBackend` -- step-level: a fitted prefill vector
  and a decode vector per capture rung, each with the convex hull it was
  fitted on; its terms are `estimated`, a step outside its hull is priced
  with the extrapolation named in their source, and a term is validated only
  inside the hull of the law's held-out steps.
- `StepGrid` / `StepGridBackend` -- step-level: a grid of measured whole
  forwards, read at a batch's equivalent homogeneous shape, answering inside a
  cell whose every corner was measured, extrapolating a decode cell past the
  KV pool and a residual past its hull, with the extrapolation named in the
  source, and refusing elsewhere.
- `ProvenanceMix` -- the run-level mixture by method and validated, and
  refusals counted by number, by fraction of steps and by fraction of
  predicted seconds. A run with nothing in
  it has no fractions and says so, rather than reporting a reassuring zero.
"""

from atom.compass.backends.base import CostBackend, Granularity
from atom.compass.backends.cost import (
    CostTerm,
    ProvenanceMix,
    StepCost,
    fold_seconds,
    fold_step,
)
from atom.compass.backends.geometry import KvGeometry, Parallelism
from atom.compass.backends.provenance import Method, Provenance, Refusal, combine
from atom.compass.backends.resolver import (
    CostRefused,
    CostSource,
    Resolution,
    Resolver,
)
from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.backends.step_grid import StepGrid, StepGridBackend
from atom.compass.backends.step_linear import StepLinearBackend, StepLinearLaw
from atom.compass.backends.step_stub import Coefficients, StepStubBackend

__all__ = [
    "BatchView",
    "Coefficients",
    "CostBackend",
    "CostRefused",
    "CostSource",
    "CostTerm",
    "Granularity",
    "KvGeometry",
    "Method",
    "Parallelism",
    "Provenance",
    "ProvenanceMix",
    "Refusal",
    "RequestShape",
    "Resolution",
    "Resolver",
    "StepCost",
    "StepGrid",
    "StepGridBackend",
    "StepLinearBackend",
    "StepLinearLaw",
    "StepStubBackend",
    "combine",
    "fold_seconds",
    "fold_step",
]
