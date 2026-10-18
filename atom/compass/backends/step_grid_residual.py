# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The step grid's residual: a factor on the lookup for what its shape cannot see.

The lookup reads a batch at its equivalent homogeneous shape, which keeps the
token count and the attention work and drops how the rows differ and the order
they run in. The residual multiplies the lookup by a factor, fitted as relative
error on replayed batches, over features a recipe computes from the rows in the
batch's own order. It is fitted apart for each grid a batch is read in, the
batch's kind (`kind`): `decode`, and `prefill_output` and `prefill_no_output`
for prefill chunks that produce output and that do not. Recipe v2's features,
with k_i the context N_KV (cached plus query) of the i-th row:

    tokens         Sum N_Q / 2^10
    query_square   Sum N_Q^2 / 2^20, prefill only: a decode row computes one
                   token, so on decode it would be tokens / 2^10
    query_context  Sum N_Q.N_KV / 2^20
    order          Sum_{i>1} |k_i - k_{i-1}| / Sum k_i: the context jump between
                   adjacent rows, as a share of the batch's context. 0 when every
                   row reads one context, small when like rows are grouped,
                   largest when they alternate.
    raggedness     B.max k / Sum k: the longest row over the mean, 1 when every
                   row reads one context.

The divisors are the recipe's normalisers, frozen with it. No feature reads a
geometry or hardware constant. A recipe is fixed code: its name and version are
written by hand, and its digest is a SHA-256 over its feature functions' source
and each kind's feature names, so editing a feature, or moving one between
kinds, without a version bump still changes the digest. A residual fitted under
another recipe name, version or digest fails loading.

A residual is a mapping, as a bundle holds it, of one or more kinds:

    {"recipe": {"name": "step-grid-residual", "version": 2, "digest": <hex>},
     "kinds": {<kind>: KIND, ...}}

    KIND = {"hull": {"equal": {<feature>: value, ...},
                     "facets": [[a_1, ..., a_k, c], ...]},
            "members": {<name>: MEMBER, ...},
            "selected": {<name>: <weight>, ...},
            "held_out": {"equal": {<axis>: value, ...},  # optional
                         "facets": [[a_1, ..., a_k, c], ...]}}

    MEMBER = {"form": "linear", "transform": "identity" | "exp",
              "intercept": z_0, "coefficients": {<feature>: z_f, ...}}
           | {"form": "band", "values": {<band>: factor, ...}}

A step of a kind the residual does not hold is refused, naming the kind; every
other step is answered by its kind's members alone. A linear member outputs its
transform of z_0 + Sum z_f.feature_f, naming every feature of its kind. A band
member outputs the factor of the batch's band (`band`): a batch of decode rows
only is in `decode rung <rung>`, `eager` when it replays no graph, and any
other in `prefill tokens <lo>-<hi>` or `<lo>+`, its token count's band over
`step_linear_fit.TOKEN_BANDS`, a member's `<lo>-` read as `<lo>+`, the top
band's spelling before; a batch in a band it holds no factor for is
refused, naming the member and the band. Only the selected members load, so an
error in one the residual does not select fails nothing, and an error in a
selected one fails loading, naming its kind and it. The factor is
Sum weight.output over the kind's selected members.

A kind's hull is its features' range over its fit, read as `hull.py` reads a
law's: `equal` holds each feature that was constant and its value, and `facets`
bound the others, in the kind's feature order, as `a . x + c <= 0` inside. A
step outside it is answered by the same members, and `factor` names how far
outside (`Hull.excess`), as the step linear law extrapolates. A step where a
member's output or the factor is not finite and positive is refused, naming
that member or the factor, and falls to the step roofline: no renormalising
over the other members, no other kind's fit, and no lookup alone.

A kind's `held_out` is the hull, as `hull` is written, of the coordinates in
their grid (`step_grid.coordinates`: B, T and c in prefill, B and c in decode)
of the batches its members were held out on. `validated` is true iff a batch
lies inside its kind's; a kind without one validates nothing.

`fit(triples, seed)` makes that mapping from (batch view, lookup seconds,
measured seconds) triples; its target is measured over lookup, whatever timed
the measurement. Each kind the triples hold is fitted on its own triples alone.
`seed` holds out `HELD_OUT` of them. The members are fitted on the rest, each
as relative error: linear over every feature of the kind, `identity` on the
target and `exp` on its log, and `band`, one factor per band the triples reach.
A feature constant over them gets coefficient zero and is named. Relative
residuals more than `MAD_K` scaled MADs from their median, floored at
`MAD_FLOOR`, are dropped and the rest refitted until the kept set settles; a
band left with no triple is named and holds no factor. The member with the
lowest median absolute relative error on the held-out triples is selected with
weight 1, a held-out triple in a band the fit holds no factor for counting as
an error of 1; the hull is that of its kept views' features, and `held_out`
that of the held-out views' coordinates, absent when they span none (too few
or flat in their varying axes). The report gives,
per kind, the held-out triples' indices and, per member, the kept and dropped
counts, the centre and scaled MAD of the relative residuals, the condition
number of the column-scaled design, the zeroed features or bands and the
held-out error. A fit holding a triple of no kind, or whose lookup or measured
time is not positive and finite, is refused; so is one where a kind's triples
hold out none or leave no more than one per coefficient, or its kept views span
no hull, naming the kind.
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
from atom.compass.backends.step_grid import _refused, coordinates
from atom.compass.backends.step_linear_fit import (
    MAD_FLOOR,
    MAD_K,
    MAD_PASSES,
    _band,
    _hull,
    _rung_key,
)

#: The share of each kind's triples `fit` holds out to select its member.
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


Feature = Callable[[Sequence[RequestShape]], Fraction]


class Recipe:
    """Each kind's feature functions of a batch's ordered rows, named,
    versioned and digested."""

    def __init__(
        self, name: str, version: int, kinds: Mapping[str, Sequence[Feature]]
    ) -> None:
        self.name, self.version = name, version
        self.features = {kind: tuple(features) for kind, features in kinds.items()}
        self.names = {
            kind: tuple(feature.__name__ for feature in features)
            for kind, features in self.features.items()
        }
        functions = dict.fromkeys(f for fs in self.features.values() for f in fs)
        source = "".join(map(inspect.getsource, functions)) + repr(self.names)
        self.digest = hashlib.sha256(source.encode()).hexdigest()

    def __call__(self, kind: str, rows: Sequence[RequestShape]) -> tuple[Fraction, ...]:
        return tuple(feature(rows) for feature in self.features[kind])


PREFILL = (tokens, query_square, query_context, order, raggedness)
RECIPE = Recipe(
    "step-grid-residual",
    2,
    {
        "decode": (tokens, query_context, order, raggedness),
        "prefill_output": PREFILL,
        "prefill_no_output": PREFILL,
    },
)


def kind(view: BatchView) -> str:
    """The grid a batch is read in, which the residual is fitted apart for;
    or a ValueError for a batch the grid refuses."""
    if view.prefill and view.decode:
        raise ValueError("a batch of prefill and decode rows is of no grid kind")
    if not view.prefill:
        return "decode"
    if view.produces_output is None:
        raise ValueError(
            "a prefill batch that does not say whether it produces output is of "
            "no grid kind"
        )
    return "prefill_output" if view.produces_output else "prefill_no_output"


def axes(kind: str) -> tuple[str, ...]:
    """The axes of a kind's grid, which `coordinates` gives."""
    return ("B", "c") if kind == "decode" else ("B", "T", "c")


class _Linear:
    """A transform of an intercept plus a coefficient on each feature."""

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

    def __call__(self, view: BatchView, x: tuple[float, ...]) -> float:
        z = self.intercept + sum(c * v for c, v in zip(self.coefficients, x))
        if self.transform == "identity":
            return z
        try:
            return math.exp(z)
        except OverflowError:
            return math.inf


def band(view: BatchView) -> str:
    """A decode batch's rung, or any other batch's token band."""
    if view.prefill:
        return _band("prefill", (1, sum_tokens(view.requests)))
    return _band(_rung_key(view.capture_rung), ())


def _spelled(key) -> str:
    """A band as `band` spells it: a residual fitted before the top prefill band
    was spelled `<lo>+` holds it as `<lo>-`."""
    key = str(key)
    return key[:-1] + "+" if key.endswith("-") else key


class _Band:
    """A factor per band."""

    def __init__(self, member: Mapping, features: tuple[str, ...]) -> None:
        self.values = {_spelled(k): float(v) for k, v in member["values"].items()}
        if len(self.values) != len(member["values"]):
            raise ValueError("a band held as both <lo>- and <lo>+")
        if not self.values:
            raise ValueError("no band")
        if any(map(_bad, self.values.values())):
            raise ValueError("a band factor that is not positive and finite")

    def __call__(self, view: BatchView, x: tuple[float, ...]) -> float:
        key = band(view)
        if key not in self.values:
            raise LookupError(f"no band {key}")
        return self.values[key]


FORMS = {"linear": _Linear, "band": _Band}


def _bad(value: float) -> bool:
    return not (math.isfinite(value) and value > 0.0)


class _Kind:
    """One kind's selected members, their weights, and the hull they were
    fitted on."""

    def __init__(self, kind: str, fitted: Mapping) -> None:
        self.kind, self.names = kind, RECIPE.names[kind]
        self.hull = Hull(f"the residual's {kind} hull", self.names, fitted["hull"])
        held = fitted.get("held_out")
        self.held_out = held and Hull(
            f"the residual's {kind} held-out hull", axes(kind), held
        )
        if not fitted["selected"]:
            raise ValueError(f"the residual selects no {kind} member")
        self.members = []
        for name, weight in fitted["selected"].items():
            try:
                member = fitted["members"][name]
                if _bad(float(weight)):
                    raise ValueError(f"weight {weight} is not positive and finite")
                if member["form"] not in FORMS:
                    raise ValueError(
                        f"form {member['form']!r} is not one of {list(FORMS)}"
                    )
                form = FORMS[member["form"]](member, self.names)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"the residual's {kind} member {name} does not load: "
                    f"{type(error).__name__}: {error}"
                ) from error
            self.members.append((name, float(weight), form))

    def factor(self, view: BatchView) -> tuple[float, str, str]:
        x = tuple(float(v) for v in RECIPE(self.kind, view.requests))
        past = ""
        if not self.hull.contains(x):
            past = f"{self.hull.excess(x):.3g} past the residual's {self.kind} hull"
        outputs = []
        for name, weight, member in self.members:
            try:
                output = member(view, x)
            except LookupError as error:
                raise _refused(
                    view, f"the residual's {self.kind} member {name} has {error}"
                )
            if _bad(output):
                raise _refused(
                    view,
                    f"the residual's {self.kind} member {name} gives {output!r}, "
                    "which is not a positive finite factor",
                )
            outputs.append((name, weight, output))
        factor = sum(weight * output for _, weight, output in outputs)
        if _bad(factor):
            raise _refused(
                view,
                f"the residual's {self.kind} factor is {factor!r}, which is not a "
                "positive finite factor",
            )
        summed = " + ".join(f"{w!r} x {name} {out!r}" for name, w, out in outputs)
        return factor, summed, past


class Residual:
    """Each fitted kind's selected members, their weights, and its hull."""

    def __init__(self, residual: Mapping) -> None:
        fitted = residual["recipe"]
        theirs = (fitted["name"], fitted["version"], fitted["digest"])
        ours = (RECIPE.name, RECIPE.version, RECIPE.digest)
        if theirs != ours:
            raise ValueError(
                "the residual was fitted under recipe {} v{} digest {}, and this "
                "code's is {} v{} digest {}".format(*theirs, *ours)
            )
        kinds = residual["kinds"]
        if not kinds or not set(kinds) <= set(RECIPE.names):
            raise ValueError(
                f"the residual fits the kinds {sorted(kinds)}, and the grid's "
                f"are one or more of {list(RECIPE.names)}"
            )
        self.kinds = {k: _Kind(k, kinds[k]) for k in RECIPE.names if k in kinds}

    def factor(self, view: BatchView) -> tuple[float, str, str]:
        """The factor for this batch's rows under its kind's fit, how it was
        summed, and how far past its kind's hull, '' inside; or a refusal."""
        try:
            k = kind(view)
        except ValueError as error:
            raise _refused(view, str(error)) from None
        if k not in self.kinds:
            raise _refused(view, f"the residual fits no {k} step")
        return self.kinds[k].factor(view)

    def validated(self, view: BatchView, at: Sequence[Fraction]) -> bool:
        """Whether a batch this residual answers, at `at` in its grid, lies
        inside its kind's held-out hull."""
        held = self.kinds[kind(view)].held_out
        return bool(held and held.contains(tuple(map(float, at))))

    def describe(self) -> str:
        kinds = "; ".join(
            f"{k}: {', '.join(name for name, _, _ in fitted.members)}"
            for k, fitted in self.kinds.items()
        )
        return f"recipe {RECIPE.name} v{RECIPE.version}, members {kinds}"


def _output(x: np.ndarray, z: np.ndarray, transform: str) -> np.ndarray:
    with np.errstate(over="ignore"):
        return x @ z if transform == "identity" else np.exp(x @ z)


def _solve(x: np.ndarray, y: np.ndarray, transform: str, vary: list[int]) -> dict:
    """Columns `vary` of `x` fitted to targets `y`, MAD-dropped until settled."""
    a, b = (
        (x / y[:, None], np.ones(len(y))) if transform == "identity" else (x, np.log(y))
    )
    keep = np.ones(len(y), dtype=bool)
    for passes in range(1, MAD_PASSES + 1):
        cols = a[keep][:, vary]
        norms = np.linalg.norm(cols, axis=0)
        # A column no kept row reaches (a band every triple of which was
        # dropped) sits out at zero.
        solved = np.array(vary)[norms > 0]
        cols, norms = cols[:, norms > 0], norms[norms > 0]
        z = np.zeros(x.shape[1])
        z[solved] = np.linalg.lstsq(cols / norms, b[keep], rcond=None)[0] / norms
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
        "zero": [j for j in range(x.shape[1]) if j not in solved],
    }


def _fit(
    kind: str, triples: Sequence[tuple[BatchView, float, float]], seed: int
) -> tuple[dict, dict]:
    """One kind's mapping, fitted on its own triples, and its report."""
    from scipy.spatial import QhullError

    names = RECIPE.names[kind]
    x = np.array(
        [[1.0, *map(float, RECIPE(kind, view.requests))] for view, _, _ in triples]
    )
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
    bands = sorted({band(triples[i][0]) for i in dev})
    onehot = np.array([[float(band(v) == b) for b in bands] for v, _, _ in triples])
    linear = [0] + [j for j in range(1, x.shape[1]) if np.ptp(x[dev, j]) > 0]
    members, solved, report = {}, {}, {"held_out": held.tolist(), "members": {}}
    for name, design, columns, vary in (
        ("identity", x, ("intercept", *names), linear),
        ("exp", x, ("intercept", *names), linear),
        ("band", onehot, bands, list(range(len(bands)))),
    ):
        transform = "exp" if name == "exp" else "identity"
        s = solved[name] = _solve(design[dev], y[dev], transform, vary)
        error = np.abs(_output(design[held], s["z"], transform) / y[held] - 1.0)
        z = [float(v) for v in s["z"]]
        members[name] = (
            {"form": "band", "values": {b: v for b, v in zip(bands, z) if v}}
            if name == "band"
            else {
                "form": "linear",
                "transform": transform,
                "intercept": z[0],
                "coefficients": dict(zip(names, z[1:])),
            }
        )
        report["members"][name] = {
            **{k: v for k, v in s.items() if k not in ("z", "keep", "zero")},
            "zero": [columns[j] for j in s["zero"]],
            "held_out_error": float(np.median(error)),
        }
    best = min(members, key=lambda t: report["members"][t]["held_out_error"])
    kept = x[dev[solved[best]["keep"]], 1:]
    try:
        hull = _span(kept, names)
    except QhullError as e:
        raise ValueError(
            f"the kept views' features span no hull: {str(e).splitlines()[0]}"
        ) from e
    out = {"hull": hull, "members": members, "selected": {best: 1.0}}
    shapes = [coordinates(triples[i][0].requests, kind != "decode") for i in held]
    try:
        out["held_out"] = _span(np.array(shapes, dtype=np.float64), axes(kind))
    except QhullError:
        pass
    return out, report


def _span(points: np.ndarray, names: tuple[str, ...]) -> dict:
    """`_hull` of `points`, a constant coordinate kept as the float it is;
    `_hull` writes it as an integer count."""
    hull = _hull(points, names)
    hull["equal"] = {f: float(points[0, names.index(f)]) for f in hull["equal"]}
    return hull


def fit(
    triples: Sequence[tuple[BatchView, float, float]], seed: int
) -> tuple[dict, dict]:
    """The residual mapping `Residual` reads, each kind fitted on its own
    triples, and its report by kind."""
    bad = [i for i, (_, *times) in enumerate(triples) if any(map(_bad, times))]
    if bad:
        raise ValueError(f"triples {bad} have a time that is not positive and finite")
    indices = {}
    for i, (view, _, _) in enumerate(triples):
        try:
            indices.setdefault(kind(view), []).append(i)
        except ValueError as error:
            raise ValueError(f"triple {i}: {error}") from None
    kinds, report = {}, {}
    for k in [k for k in RECIPE.names if k in indices]:
        own = indices[k]
        try:
            kinds[k], report[k] = _fit(k, [triples[i] for i in own], seed)
        except ValueError as error:
            raise ValueError(f"the {k} fit: {error}") from error
        report[k]["held_out"] = [own[i] for i in report[k]["held_out"]]
    recipe = {"name": RECIPE.name, "version": RECIPE.version, "digest": RECIPE.digest}
    return {"recipe": recipe, "kinds": kinds}, report
