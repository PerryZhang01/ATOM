# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Tier a: one rank's step priced by a fitted linear law.

A step is prefill or decode, never both. Prefill has one coefficient vector,
decode one per capture rung, each over the features `ShapeStubBackend` reads:

    prefill:        [1, tokens, Sum N_Q^2, Sum N_Q.(N_KV - N_Q)]
    decode, rung r: [1, Sum N_KV, r.max N_KV - Sum N_KV]

A prefill vector may add `history`, Sum (N_KV - N_Q): the KV a chunk reads
on its cached history, which a short chunk on a deep history pays apart from
its attention arithmetic.

A decode step that replays no graph (an eager runner, or a batch wider than
the widest captured size) is the rung named `eager`, whose padding is zero.

Each vector carries the convex hull of the features it was fitted on. A step
outside it is priced by the same vector and its terms are `extrapolated`, with
a detail naming the hull; it is not refused. A step the law has no vector for
is refused: prefill when the law has no prefill vector, a decode rung with
none, or a step mixing prefill and decode rows.

The law is a mapping, as a run file holds it, `prefill` optional:

    {"provenance": "<what it was fitted on>",
     "prefill": VECTOR,
     "decode": {"eager": VECTOR, "1": VECTOR, "2": VECTOR, ...}}

    VECTOR = {"coefficients": {<feature>: seconds, ...},
              "hull": {"equal": {<feature>: value, ...}, "facets": [[a_1, ..., a_k, c], ...]}}

`coefficients` names every feature of its kind, `step` for the intercept.
`equal` holds each feature that was constant over the fit and the value a step
must have; `facets` bound the remaining non-intercept features, in the order
above, as `a . x + c <= 0` inside: the rows of qhull's `ConvexHull.equations`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from atom.compass.backends.base import CostBackend, Tier
from atom.compass.backends.cost import CostTerm, StepCost
from atom.compass.backends.ladder import CostRefused
from atom.compass.backends.provenance import Provenance, Refusal, Species
from atom.compass.backends.shape import (
    BatchView,
    sum_context,
    sum_query_cached,
    sum_query_square,
    sum_tokens,
)

PREFILL = ("step", "tokens", "query_square", "query_cached")
HISTORY = PREFILL + ("history",)
DECODE = ("step", "context", "graph_padding")
EAGER = "eager"
SOURCE = "coarse-law"
#: A point on a facet evaluates to rounding error, relative to its magnitudes.
_HULL_TOLERANCE = 1e-9


class _Vector:
    """One kind's (and rung's) coefficients and the hull they were fitted on."""

    def __init__(self, name: str, features: tuple[str, ...], spec: Mapping) -> None:
        self.name = name
        coefficients = spec["coefficients"]
        if set(coefficients) != set(features):
            raise ValueError(
                f"the {name} vector has coefficients for {sorted(coefficients)}, "
                f"and its features are {list(features)}"
            )
        for feature, value in coefficients.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"the {name} vector's {feature} coefficient is {value}; a term "
                    "is a non-negative duration"
                )
        self.coefficients = tuple(float(coefficients[f]) for f in features)
        # The intercept, first in `features`, is not a hull coordinate.
        self.hull = Hull(f"the {name} hull", features[1:], spec["hull"])
        self.features = features


class Hull:
    """The convex hull of the features a fit ranged over.

    `equal` maps each feature that was constant over the fit to its value;
    `facets` bound the others, in `features` order, as `a . x + c <= 0` inside.
    `label` names the hull in every load error.
    """

    def __init__(self, label: str, features: tuple[str, ...], hull: Mapping) -> None:
        equal = hull.get("equal", {})
        if not set(equal) <= set(features):
            raise ValueError(
                f"{label} checks {sorted(equal)} for equality, which "
                f"are not all among its features {list(features)}"
            )
        self.equal = {features.index(f): value for f, value in equal.items()}
        self.vary = [i for i in range(len(features)) if i not in self.equal]
        width = len(self.vary) + 1
        if any(len(row) != width for row in hull["facets"]):
            raise ValueError(
                f"{label} has a facet not {width} wide: one entry per "
                f"varying feature {[features[i] for i in self.vary]}, then the offset"
            )
        self.facets = np.asarray(hull["facets"], dtype=np.float64).reshape(-1, width)
        if self.vary and not len(self.facets):
            raise ValueError(
                f"{label} bounds {len(self.vary)} varying features with no "
                "facets, which is all of space"
            )
        if not np.isfinite(self.facets).all():
            raise ValueError(f"{label} has a facet that is not finite")

    def contains(self, point: tuple[float, ...]) -> bool:
        if any(point[i] != value for i, value in self.equal.items()):
            return False
        x = np.array([point[i] for i in self.vary], dtype=np.float64)
        normals, offsets = self.facets[:, :-1], self.facets[:, -1]
        scale = np.abs(normals) @ np.abs(x) + np.abs(offsets)
        return bool(np.all(normals @ x + offsets <= _HULL_TOLERANCE * scale))


class CoarseLaw:
    """A prefill vector, a decode vector per rung, each with its hull."""

    def __init__(self, law: Mapping) -> None:
        self.provenance = str(law.get("provenance", "")).strip()
        if not self.provenance:
            raise ValueError("a law states what it was fitted on, under provenance")
        self.prefill = None
        if "prefill" in law:
            prefill = law["prefill"]
            features = HISTORY if "history" in prefill["coefficients"] else PREFILL
            self.prefill = _Vector("prefill", features, prefill)
        self.decode = {
            None if rung == EAGER else int(rung): _Vector(
                f"decode rung {rung}", DECODE, spec
            )
            for rung, spec in law["decode"].items()
        }


class CoarseBackend(CostBackend):
    """Prices a step as its features times the law's coefficients, one term each."""

    def __init__(self, law: CoarseLaw) -> None:
        self.law = law

    @property
    def tier(self) -> Tier:
        return Tier.COARSE

    def _refuse(self, view: BatchView, reason: str) -> CostRefused:
        return CostRefused(view, [Refusal(SOURCE, reason)])

    def estimate(self, batch_view: BatchView) -> StepCost:
        if not isinstance(batch_view, BatchView):
            raise TypeError(
                f"this backend prices a BatchView, not {type(batch_view).__name__}"
            )
        prefill, decode = batch_view.prefill, batch_view.decode
        if prefill and decode:
            raise self._refuse(
                batch_view,
                f"a step of {len(prefill)} prefill and {len(decode)} decode rows; "
                "the law prices a step of one kind",
            )
        if prefill:
            vector, kind = self.law.prefill, "prefill"
            if vector is None:
                raise self._refuse(batch_view, "no prefill vector")
            counts = (
                1,
                sum_tokens(prefill),
                sum_query_square(prefill),
                sum_query_cached(prefill),
            )
            if len(vector.features) == len(HISTORY):
                counts += (sum_context(prefill) - sum_tokens(prefill),)
        else:
            rung, kind = batch_view.capture_rung, "decode"
            vector = self.law.decode.get(rung)
            if vector is None:
                raise self._refuse(
                    batch_view,
                    f"no decode vector for rung {EAGER if rung is None else rung}",
                )
            counts = (1, sum_context(decode), batch_view.graph_padding)
        detail = self.law.provenance
        species = Species.FITTED
        if not vector.hull.contains(counts[1:]):
            species = Species.EXTRAPOLATED
            detail = f"{detail}; outside the {vector.name} hull"
        return StepCost(
            [
                CostTerm(
                    f"{kind}.{feature}",
                    count * coefficient,
                    Provenance(
                        species, f"{coefficient:g} s x {count}; {detail}", SOURCE
                    ),
                )
                for feature, count, coefficient in zip(
                    vector.features, counts, vector.coefficients
                )
            ]
        )

    def describe(self) -> str:
        return f"tier-a coarse law; {self.law.provenance}"
