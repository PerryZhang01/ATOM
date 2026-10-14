# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The step grid's residual: a factor on the lookup for what its shape cannot see.

The lookup reads a batch at its equivalent homogeneous shape, which keeps the
token count and the attention work and drops how the rows differ and the order
they run in. The residual multiplies the lookup by a factor, fitted as relative
error on replayed batches, over features a recipe computes from the rows in the
batch's own order. Recipe v1's, with k_i the context N_KV (cached plus query)
of the i-th row:

    tokens         Sum N_Q / 2^10
    query_square   Sum N_Q^2 / 2^20
    query_context  Sum N_Q.N_KV / 2^20
    order          Sum_{i>1} |k_i - k_{i-1}| / Sum k_i: the context jump between
                   adjacent rows, as a share of the batch's context. 0 when every
                   row reads one context, small when like rows are grouped,
                   largest when they alternate.
    raggedness     B.max k / Sum k: the longest row over the mean, 1 when every
                   row reads one context.

The divisors are the recipe's normalisers, frozen with it. No feature reads a
geometry or hardware constant. A recipe is fixed code: its name and version are
written by hand, and its digest is a SHA-256 over its feature functions'
source, so editing a feature without a version bump still changes the digest.
A residual fitted under another recipe name, version or digest fails loading.

A residual is a mapping, as a bundle holds it:

    {"recipe": {"name": "step-grid-residual", "version": 1, "digest": <hex>},
     "hull": {"equal": {<feature>: value, ...}, "facets": [[a_1, ..., a_k, c], ...]},
     "members": {<name>: MEMBER, ...},
     "selected": {<name>: <weight>, ...}}

    MEMBER = {"form": "linear", "transform": "identity" | "exp",
              "intercept": z_0, "coefficients": {<feature>: z_f, ...}}

A linear member outputs its transform of z_0 + Sum z_f.feature_f, naming every
feature of the recipe; `form` leaves room for other forms beside it. Only the
selected members load, so an error in one the residual does not select fails
nothing, and an error in a selected one fails loading, naming it. The factor is
Sum weight.output over the selected members.

The hull is the features' range over the fit, read as `hull.py` reads a
law's: `equal` holds each feature that was constant and its value, and `facets`
bound the others, in recipe order, as `a . x + c <= 0` inside. A step outside
it is refused, naming the hull; so is a step where a member's output or the
factor is not finite and positive, naming that member or the factor. Nothing
else answers: no renormalising over the other members, and no lookup alone.
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction

from atom.compass.backends.hull import Hull
from atom.compass.backends.shape import (
    BatchView,
    RequestShape,
    sum_context,
    sum_query_context,
    sum_query_square,
    sum_tokens,
)
from atom.compass.backends.step_grid import _refused


def tokens(rows: Sequence[RequestShape]) -> Fraction:
    return Fraction(sum_tokens(rows), 2**10)


def query_square(rows: Sequence[RequestShape]) -> Fraction:
    return Fraction(sum_query_square(rows), 2**20)


def query_context(rows: Sequence[RequestShape]) -> Fraction:
    return Fraction(sum_query_context(rows), 2**20)


def order(rows: Sequence[RequestShape]) -> Fraction:
    jumps = sum(
        abs(b.context_tokens - a.context_tokens) for a, b in itertools.pairwise(rows)
    )
    return Fraction(jumps, sum_context(rows))


def raggedness(rows: Sequence[RequestShape]) -> Fraction:
    longest = max(row.context_tokens for row in rows)
    return Fraction(len(rows) * longest, sum_context(rows))


class Recipe:
    """Feature functions of a batch's ordered rows, named, versioned and digested."""

    def __init__(
        self,
        name: str,
        version: int,
        features: Sequence[Callable[[Sequence[RequestShape]], Fraction]],
    ) -> None:
        self.name, self.version, self.features = name, version, tuple(features)
        self.names = tuple(feature.__name__ for feature in self.features)
        source = "".join(inspect.getsource(feature) for feature in self.features)
        self.digest = hashlib.sha256(source.encode()).hexdigest()

    def __call__(self, rows: Sequence[RequestShape]) -> tuple[Fraction, ...]:
        return tuple(feature(rows) for feature in self.features)


V1 = Recipe(
    "step-grid-residual", 1, (tokens, query_square, query_context, order, raggedness)
)


class _Linear:
    """A transform of an intercept plus a coefficient on each recipe feature."""

    def __init__(self, member: Mapping, features: tuple[str, ...]) -> None:
        self.transform = member["transform"]
        if self.transform not in ("identity", "exp"):
            raise ValueError(
                f"transform {self.transform!r} is not one of ['identity', 'exp']"
            )
        coefficients = member["coefficients"]
        if set(coefficients) != set(features):
            raise ValueError(
                f"coefficients for {sorted(coefficients)}, and the recipe's "
                f"features are {list(features)}"
            )
        values = [float(member["intercept"])]
        values += [float(coefficients[f]) for f in features]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("an intercept or coefficient that is not finite")
        self.intercept, *self.coefficients = values

    def __call__(self, x: tuple[float, ...]) -> float:
        z = self.intercept + sum(c * v for c, v in zip(self.coefficients, x))
        if self.transform == "identity":
            return z
        try:
            return math.exp(z)
        except OverflowError:
            return math.inf


FORMS = {"linear": _Linear}


def _bad(value: float) -> bool:
    return not (math.isfinite(value) and value > 0.0)


class Residual:
    """The selected members, their weights, and the hull they were fitted on."""

    def __init__(self, residual: Mapping) -> None:
        self.recipe = V1
        fitted = residual["recipe"]
        theirs = (fitted["name"], fitted["version"], fitted["digest"])
        ours = (V1.name, V1.version, V1.digest)
        if theirs != ours:
            raise ValueError(
                "the residual was fitted under recipe {} v{} digest {}, and this "
                "code's is {} v{} digest {}".format(*theirs, *ours)
            )
        self.hull = Hull("the residual's hull", V1.names, residual["hull"])
        if not residual["selected"]:
            raise ValueError("the residual selects no member")
        self.members = []
        for name, weight in residual["selected"].items():
            try:
                member = residual["members"][name]
                if _bad(float(weight)):
                    raise ValueError(f"weight {weight} is not positive and finite")
                if member["form"] not in FORMS:
                    raise ValueError(
                        f"form {member['form']!r} is not one of {list(FORMS)}"
                    )
                form = FORMS[member["form"]](member, V1.names)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"the residual's member {name} does not load: "
                    f"{type(error).__name__}: {error}"
                ) from error
            self.members.append((name, float(weight), form))

    def factor(self, view: BatchView) -> tuple[float, str]:
        """The factor for this batch's rows and how it was summed; or a refusal."""
        x = tuple(float(v) for v in self.recipe(view.requests))
        if not self.hull.contains(x):
            named = ", ".join(f"{f}={v:g}" for f, v in zip(self.recipe.names, x))
            raise _refused(view, f"the step at {named} is outside the residual's hull")
        outputs = []
        for name, weight, member in self.members:
            output = member(x)
            if _bad(output):
                raise _refused(
                    view,
                    f"the residual's member {name} gives {output!r}, which is not "
                    "a positive finite factor",
                )
            outputs.append((name, weight, output))
        factor = sum(weight * output for _, weight, output in outputs)
        if _bad(factor):
            raise _refused(
                view,
                f"the residual's factor is {factor!r}, which is not a positive "
                "finite factor",
            )
        summed = " + ".join(f"{w!r} x {name} {out!r}" for name, w, out in outputs)
        return factor, summed

    def describe(self) -> str:
        names = ", ".join(name for name, _, _ in self.members)
        return f"recipe {self.recipe.name} v{self.recipe.version}, members {names}"
