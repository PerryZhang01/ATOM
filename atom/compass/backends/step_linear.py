# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The step linear law: one rank's step priced by a fitted linear law.

A step is prefill or decode, never both. Prefill has one coefficient vector,
decode one per capture rung, each over the features `StepStubBackend` reads:

    prefill:        [1, tokens, Sum N_Q^2, Sum N_Q.(N_KV - N_Q)]
    decode, rung r: [1, Sum N_KV, r.max N_KV - Sum N_KV]

A prefill vector may add `history`, Sum (N_KV - N_Q): the KV a chunk reads
on its cached history, which a short chunk on a deep history pays apart from
its attention arithmetic, and then `moe_rows`, the rows the data-parallel
group's MoE gather carries (`BatchView.moe_rows`), which every rank's MoE runs on.

A law prices one rank's step; the runner takes the group's step as the max
over its ranks. Its parallel layout is its `engine_config`'s, which a run
checks; a law stating `key` is refused. A backend built for a data-parallel
width above 1 names `UNIFORM_ROUTING` among every step's treatments: the MoE
segment is priced as if routing spread tokens evenly over the experts, which
no feature carries. A run refuses a law whose
`machine_spec` is not its spec's `machine_spec_digest`, naming both.

A decode vector may add `rowmod`, `rowmod(k)` over its rows in scheduled
order: the largest summed N_KV over the rows `i = r (mod k)`, with `k` the
vector's `"rowmod"` entry. It prices an order the same multiset of rows can
take at a different cost; `rowmod(1)` is Sum N_KV.

A decode step that replays no graph (an eager runner, or a batch wider than
the widest captured size) is the rung named `eager`, whose padding is zero.

Every term is `estimated`. Each vector carries the convex hull of the features
it was fitted on. A step outside it is priced by the same vector, its terms'
source is `EXTRAPOLATED` and their detail names the hull; it is not refused.
A vector may also carry, under `held_out`, the hull of the steps
`step_linear_fit` priced with a law fitted on the other records and their
errors per band: a term is validated iff its step lies inside that hull, so a
law without one prices every term unvalidated. A step the law has no vector for
is refused: prefill when the law has no prefill vector, a decode rung with
none, a step mixing prefill and decode rows, or one with image tokens.

A law fitted on an MTP record holds a draft law of its own form under `draft`.
Its terms, named `draft.<kind>.<feature>`, are tagged `draft` and priced from the
same features, on a step that produces no output too: there the draft is the
pass that fills the drafter's KV over the chunk.

The law is a mapping, as a run file holds it, `prefill` and `draft` optional:

    {"provenance": "<what it was fitted on>",
     "machine_spec": "sha256:<hex>",
     "prefill": VECTOR,
     "decode": {"eager": VECTOR, "1": VECTOR, "2": VECTOR, ...},
     "draft": {"provenance": ..., "prefill": VECTOR, "decode": {...}}}

    VECTOR = {"coefficients": {<feature>: seconds, ...},
              "hull": HULL,
              "rowmod": k,    # a decode vector with a `rowmod` coefficient only
              "held_out": {"hull": HULL, "bands": {<band>: ERRORS, ...}}}  # optional

    HULL = {"equal": {<feature>: value, ...}, "facets": [[a_1, ..., a_k, c], ...]}
    ERRORS = {"steps": n, "p10": e, "p50": e, "p90": e}, e = priced / real - 1

`coefficients` names every feature of its kind, `step` for the intercept.
`equal` holds each feature that was constant over the fit and the value a step
must have; `facets` bound the remaining non-intercept features, in the order
above, as `a . x + c <= 0` inside: the rows of qhull's `ConvexHull.equations`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from atom.compass.backends.base import CostBackend, Granularity
from atom.compass.backends.cost import CostTerm, StepCost
from atom.compass.backends.hull import Hull
from atom.compass.backends.provenance import Method, Provenance, Refusal
from atom.compass.backends.resolver import CostRefused
from atom.compass.backends.shape import (
    IMAGES,
    BatchView,
    sum_context,
    sum_query_cached,
    sum_query_square,
    sum_tokens,
)

PREFILL = ("step", "tokens", "query_square", "query_cached")
HISTORY = PREFILL + ("history",)
DECODE = ("step", "context", "graph_padding")
ROWMOD = DECODE + ("rowmod",)
EAGER = "eager"
SOURCE = "step-linear"
EXTRAPOLATED = "step-linear-extrapolated"
UNIFORM_ROUTING = "uniform routing"


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
        held = spec.get("held_out")
        self.held_out = held and Hull(
            f"the {name} held-out hull", features[1:], held["hull"]
        )
        self.features = features
        self.rowmod = int(spec["rowmod"]) if "rowmod" in features else None
        if self.rowmod is not None and self.rowmod < 1:
            raise ValueError(f"the {name} vector's rowmod modulus is {self.rowmod}")


def rowmod(context, k: int) -> int:
    """The largest summed context over rows `i = r (mod k)`, in `context`'s order."""
    return max(sum(context[r::k]) for r in range(min(k, len(context))))


class StepLinearLaw:
    """A prefill vector, a decode vector per rung, each with its hull."""

    def __init__(self, law: Mapping) -> None:
        self.fitted_on = str(law.get("provenance", "")).strip()
        if not self.fitted_on:
            raise ValueError("a law states what it was fitted on, under provenance")
        if "key" in law:
            raise ValueError(
                f"the law states key {law['key']}; its engine_config states the "
                "parallel layout it was fitted at"
            )
        self.prefill = None
        if "prefill" in law:
            prefill = law["prefill"]
            extra = [f for f in ("history", "moe_rows") if f in prefill["coefficients"]]
            self.prefill = _Vector("prefill", PREFILL + tuple(extra), prefill)
        self.decode = {
            None if rung == EAGER else int(rung): _Vector(
                f"decode rung {rung}",
                ROWMOD if "rowmod" in spec["coefficients"] else DECODE,
                spec,
            )
            for rung, spec in law["decode"].items()
        }
        self.draft = StepLinearLaw(law["draft"]) if "draft" in law else None


class StepLinearBackend(CostBackend):
    """Prices a step as its features times the law's coefficients, one term each."""

    def __init__(self, law: StepLinearLaw, dp: int = 1) -> None:
        self.law = law
        self.treatments = (UNIFORM_ROUTING,) if dp > 1 else ()

    @property
    def granularity(self) -> Granularity:
        return Granularity.STEP

    def _refuse(self, view: BatchView, reason: str) -> CostRefused:
        return CostRefused(view, [Refusal(SOURCE, reason)])

    def price(self, batch_view: BatchView) -> StepCost:
        if not isinstance(batch_view, BatchView):
            raise TypeError(
                f"this backend prices a BatchView, not {type(batch_view).__name__}"
            )
        prefill, decode = batch_view.prefill, batch_view.decode
        if batch_view.image_tokens:
            raise self._refuse(batch_view, IMAGES.format(batch_view.image_tokens))
        if prefill and decode:
            raise self._refuse(
                batch_view,
                f"a step of {len(prefill)} prefill and {len(decode)} decode rows; "
                "the law prices a step of one kind",
            )
        terms = self._terms(self.law, batch_view, "")
        if self.law.draft is not None:
            terms += self._terms(self.law.draft, batch_view, "draft ")
        return StepCost(terms, self.treatments)

    def _terms(self, law: StepLinearLaw, batch_view: BatchView, what: str) -> list:
        """`law`'s terms for the step, named and tagged as draft when `what` is."""
        prefill, decode = batch_view.prefill, batch_view.decode
        if prefill:
            vector, kind = law.prefill, "prefill"
            if vector is None:
                raise self._refuse(batch_view, f"no {what}prefill vector")
            counts = (
                1,
                sum_tokens(prefill),
                sum_query_square(prefill),
                sum_query_cached(prefill),
            )
            if "history" in vector.features:
                counts += (sum_context(prefill) - sum_tokens(prefill),)
            if "moe_rows" in vector.features:
                if batch_view.moe_rows is None:
                    raise self._refuse(batch_view, "no MoE gather rows on this step")
                counts += (batch_view.moe_rows,)
        else:
            rung, kind = batch_view.capture_rung, "decode"
            vector = law.decode.get(rung)
            if vector is None:
                raise self._refuse(
                    batch_view,
                    f"no {what}decode vector for rung {EAGER if rung is None else rung}",
                )
            counts = (1, sum_context(decode), batch_view.graph_padding)
            if vector.rowmod:
                counts += (rowmod([r.context_tokens for r in decode], vector.rowmod),)
        detail, source = law.fitted_on, SOURCE
        if not vector.hull.contains(counts[1:]):
            source = EXTRAPOLATED
            detail = f"{detail}; outside the {what}{vector.name} hull"
        validated = bool(vector.held_out and vector.held_out.contains(counts[1:]))
        return [
            CostTerm(
                f"{what.replace(' ', '.')}{kind}.{feature}",
                count * coefficient,
                Provenance(
                    Method.ESTIMATED,
                    f"{coefficient:g} s x {count}; {detail}",
                    source,
                    validated=validated,
                ),
                draft=bool(what),
            )
            for feature, count, coefficient in zip(
                vector.features, counts, vector.coefficients
            )
        ]

    def describe(self) -> str:
        return f"step linear law; {self.law.fitted_on}"
