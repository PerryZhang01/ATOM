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

`fit(triples, seed)` makes that mapping from (batch view, lookup seconds,
measured seconds) triples; its target is measured over lookup, whatever timed
the measurement. `seed` holds out `HELD_OUT` of the triples. Two linear
members are fitted on the rest over every recipe feature: `identity` on the
target, as relative error, and `exp` on its log. A feature constant over them
gets coefficient zero and is named. Relative residuals more than `MAD_K`
scaled MADs from their median, floored at `MAD_FLOOR`, are dropped and the
rest refitted until the kept set settles. The member with the lower median
absolute relative error on the held-out triples is selected with weight 1;
the hull is that of its kept views' features. The report gives the held-out
triples' indices and, per member, the kept and dropped counts, the centre and
scaled MAD of the relative residuals, the condition number of the
column-scaled design, the zeroed features and the held-out error. A fit whose
triples hold out none or leave no more than one per coefficient, whose
lookup or measured time is not positive and finite, or whose kept views span
no hull is refused.
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction

import numpy as np

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
from atom.compass.backends.step_linear_fit import MAD_FLOOR, MAD_K, MAD_PASSES, _hull

#: The share of the triples `fit` holds out to select its member.
HELD_OUT = 0.25


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


def _output(x: np.ndarray, z: np.ndarray, transform: str) -> np.ndarray:
    with np.errstate(over="ignore"):
        return x @ z if transform == "identity" else np.exp(x @ z)


def _solve(x: np.ndarray, y: np.ndarray, transform: str) -> dict:
    """One linear member on `x` (intercept first) for targets `y`, MAD-dropped until settled."""
    vary = [0] + [j for j in range(1, x.shape[1]) if np.ptp(x[:, j]) > 0]
    a, b = (
        (x / y[:, None], np.ones(len(y))) if transform == "identity" else (x, np.log(y))
    )
    keep = np.ones(len(y), dtype=bool)
    for passes in range(1, MAD_PASSES + 1):
        cols = a[keep][:, vary]
        norms = np.linalg.norm(cols, axis=0)
        z = np.zeros(x.shape[1])
        z[vary] = np.linalg.lstsq(cols / norms, b[keep], rcond=None)[0] / norms
        residual = _output(x, z, transform) / y - 1.0
        centre = np.median(residual)
        mad = 1.4826 * np.median(np.abs(residual - centre))
        moved = np.abs(residual - centre) <= max(MAD_K * mad, MAD_FLOOR)
        if (moved == keep).all() or passes == MAD_PASSES:
            break
        keep = moved
    return {
        "z": z,
        "keep": keep,
        "samples": int(keep.sum()),
        "dropped": int((~keep).sum()),
        "centre": float(centre),
        "mad": float(mad),
        "condition": float(np.linalg.cond(cols / norms)),
        "zero": [V1.names[j - 1] for j in range(1, x.shape[1]) if j not in vary],
    }


def fit(
    triples: Sequence[tuple[BatchView, float, float]], seed: int
) -> tuple[dict, dict]:
    """The residual mapping `Residual` reads, fitted on `triples`, and its report."""
    from scipy.spatial import QhullError

    bad = [i for i, (_, *times) in enumerate(triples) if any(map(_bad, times))]
    if bad:
        raise ValueError(f"triples {bad} have a time that is not positive and finite")
    x = np.array([[1.0, *map(float, V1(view.requests))] for view, _, _ in triples])
    y = np.array([measured / lookup for _, lookup, measured in triples])
    held = np.sort(
        np.random.default_rng(seed).permutation(len(y))[: int(len(y) * HELD_OUT)]
    )
    dev = np.setdiff1d(np.arange(len(y)), held)
    if not len(held) or len(dev) <= x.shape[1]:
        raise ValueError(
            f"{len(y)} triples hold out {len(held)} and leave {len(dev)} to fit "
            f"{x.shape[1]} coefficients"
        )
    members, solved, report = {}, {}, {"held_out": held.tolist(), "members": {}}
    for transform in ("identity", "exp"):
        s = solved[transform] = _solve(x[dev], y[dev], transform)
        error = np.abs(_output(x[held], s["z"], transform) / y[held] - 1.0)
        members[transform] = {
            "form": "linear",
            "transform": transform,
            "intercept": float(s["z"][0]),
            "coefficients": dict(zip(V1.names, map(float, s["z"][1:]))),
        }
        report["members"][transform] = {
            **{k: v for k, v in s.items() if k not in ("z", "keep")},
            "held_out_error": float(np.median(error)),
        }
    best = min(members, key=lambda t: report["members"][t]["held_out_error"])
    kept = x[dev[solved[best]["keep"]], 1:]
    try:
        hull = _hull(kept, V1.names)
    except QhullError as e:
        raise ValueError(
            f"the kept views' features span no hull: {str(e).splitlines()[0]}"
        ) from e
    # `_hull` writes a constant feature as an integer count; these are fractions.
    hull["equal"] = {f: float(kept[0, V1.names.index(f)]) for f in hull["equal"]}
    recipe = {"name": V1.name, "version": V1.version, "digest": V1.digest}
    mapping = {
        "recipe": recipe,
        "hull": hull,
        "members": members,
        "selected": {best: 1.0},
    }
    return mapping, report
