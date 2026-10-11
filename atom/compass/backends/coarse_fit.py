# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Tier a's law fitted from real step records.

    python -m atom.compass.backends.coarse_fit RECORD... --out LAW [--form FORM]

Each RECORD is a real run's step record (`steps/dp0.jsonl`, written with
`ATOM_COMPASS_PARITY_RECORD`). A step's label is its period: `t_enter_ns` of
the step to `t_enter_ns` of the next, kept only where the next batch holds a
request this one left unfinished, so the engine went straight from one forward
to the next. The host time between forwards lands in the intercepts. A step
whose period may hold an idle gap and a step mixing prefill and decode rows
are counted and not fitted. The step after an idle gap is fitted: its period
starts at its own forward, after the gap.

Each equation is divided by its own label and the coefficients are bound to be
non-negative. Steps whose relative residual is more than `MAD_K` scaled MADs
from the median are dropped and the rest refitted, until the kept set settles.
The dropped steps that lead a record's steps of one vector, each slower than
the law, are its warm steps (shape compiles and first touches), and so is a
dropped step anywhere that ran more than `WARM_FACTOR` times its price, such as
the first short chunk on a deep history: counted with their seconds apart from
the dropped ones, never charged. A feature with no variance gets coefficient
zero and is named. Each vector reports its samples,
warm and dropped counts, the condition number of its column-scaled design and
the convex hull of its fitted steps, which `CoarseLaw` reads. A vector with
no more fitted steps than coefficients, or whose steps span no hull, is left
out and named under `unfitted`; `CoarseBackend` refuses its steps.

Forms (`FORMS`): `law`, the features `CoarseLaw` always prices; `history`,
prefill plus `Sum (N_KV - N_Q)`, the KV a short chunk reads on a deep history;
`shared`, decode with the context and padding slopes shared by every rung and
an intercept per rung.

With two or more records it also prints, per form, the error on each record of
a law fitted on the others, warm steps left out: per kind and rung, or prefill
token band, the p10, p50 and p90 of `predicted / measured - 1`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

from atom.compass.backends.coarse import DECODE, EAGER, HISTORY, PREFILL
from atom.compass.backends.shape import (
    BatchView,
    RequestShape,
    sum_context,
    sum_query_cached,
    sum_query_square,
    sum_tokens,
)

FORMS = ("law", "history", "shared")
MAD_K = 3.5
#: A relative residual this small is never an outlier, however tight the rest.
MAD_FLOOR = 0.01
MAD_PASSES = 20
#: A step that ran this many times its price is a one-off first touch, not scatter.
WARM_FACTOR = 5.0
TOKEN_BANDS = (512, 2048, 8192)


def _rung_key(rung: int | None) -> str:
    return EAGER if rung is None else str(rung)


def samples(record, form: str = "law") -> tuple[list[tuple], dict]:
    """One record's fittable steps as (vector, features, seconds, step), and what was cut.

    `vector` is `prefill` or a decode rung's key; features are in `CoarseLaw`'s
    order, plus `history` for that form.
    """
    steps = [json.loads(line) for line in Path(record).read_text().splitlines()]
    steps = [s for s in steps if not s["is_dummy_run"]]
    if not steps or "t_enter_ns" not in steps[0]:
        raise ValueError(
            f"{record} has no timed steps; a fit needs a real run's record"
        )
    cut = {"idle": 0, "mixed": 0, "last": 1}
    rows = []
    for step, nxt in pairwise(steps):
        idle = not {r[0] for r in step["batch"]} & {r[0] for r in nxt["batch"]}
        view = BatchView(
            tuple(RequestShape(q, c, bool(d)) for q, c, d in step["rows"]),
            capture_rung=step["rung"],
        )
        if idle:
            cut["idle"] += 1
        elif view.prefill and view.decode:
            cut["mixed"] += 1
        else:
            if view.prefill:
                r = view.prefill
                key = "prefill"
                counts = (1, sum_tokens(r), sum_query_square(r), sum_query_cached(r))
                if form == "history":
                    counts += (sum_context(r) - sum_tokens(r),)
            else:
                key = _rung_key(view.capture_rung)
                counts = (1, sum_context(view.decode), view.graph_padding)
            period = (nxt["t_enter_ns"] - step["t_enter_ns"]) * 1e-9
            rows.append((key, counts, period, step["step"]))
    return rows, cut


def _solve(x: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
    """Non-negative least squares on relative error, MAD-dropped until settled."""
    from scipy.optimize import nnls

    a = x / y[:, None]
    # An intercept is kept whatever its variance; any other constant column is not.
    vary = [
        j
        for j in range(x.shape[1])
        if names[j].startswith("step") or np.ptp(x[:, j]) > 0
    ]
    keep = np.ones(len(y), dtype=bool)
    # Refitted until the kept set stops moving, so a step an outlier's pull
    # dropped on the first pass comes back once the outlier is gone.
    for passes in range(1, MAD_PASSES + 1):
        cols = a[keep][:, vary]
        norms = np.linalg.norm(cols, axis=0)
        coef = np.zeros(x.shape[1])
        coef[vary] = nnls(cols / norms, np.ones(int(keep.sum())))[0] / norms
        residual = a @ coef - 1.0
        centre = np.median(residual)
        mad = 1.4826 * np.median(np.abs(residual - centre))
        moved = np.abs(residual - centre) <= max(MAD_K * mad, MAD_FLOOR)
        if (moved == keep).all() or passes == MAD_PASSES:
            break
        keep = moved
    return {
        "coef": coef,
        "keep": keep,
        "residual": residual,
        "condition": float(np.linalg.cond(cols / norms)),
        "zero": [names[j] for j in range(x.shape[1]) if j not in vary],
    }


def _hull(points: np.ndarray, names: tuple[str, ...]) -> dict:
    """`CoarseLaw`'s hull over the non-intercept features of the fitted steps."""
    from scipy.spatial import ConvexHull

    equal = {
        names[j]: int(points[0, j])
        for j in range(points.shape[1])
        if np.ptp(points[:, j]) == 0
    }
    vary = [j for j in range(points.shape[1]) if names[j] not in equal]
    x = points[:, vary]
    lo, hi = x.min(axis=0), x.max(axis=0)
    if len(vary) == 1:
        facets = [[-1.0, float(lo[0])], [1.0, -float(hi[0])]]
    elif vary:
        # Hulled in the unit box, where qhull's tolerances suit every feature.
        eq = ConvexHull((x - lo) / (hi - lo)).equations
        normals = eq[:, :-1] / (hi - lo)
        facets = np.column_stack([normals, eq[:, -1] - normals @ lo]).tolist()
    else:
        facets = []
    return {"equal": equal, "facets": facets}


def fit(records, form: str = "law") -> dict:
    """The law over `records`, as the mapping `CoarseLaw` reads."""
    from scipy.spatial import QhullError

    if form not in FORMS:
        raise ValueError(f"form {form!r} is not one of {FORMS}")
    rows, cut = [], {"warm": 0, "warm_s": 0.0}
    for i, record in enumerate(records):
        got, c = samples(record, form)
        rows += [(i, *r) for r in got]
        for k, v in c.items():
            cut[k] = cut.get(k, 0) + v
    digests = [
        f"{Path(r).name}:{hashlib.sha256(Path(r).read_bytes()).hexdigest()[:12]}"
        for r in records
    ]
    law = {
        "provenance": f"coarse_fit {form} on {len(records)} real records "
        f"({', '.join(digests)})",
        "form": form,
        "cut": cut,
        "warm_steps": [[] for _ in records],
        "unfitted": {},
        "decode": {},
    }
    keys = sorted(
        {r[1] for r in rows},
        key=lambda k: (k != "prefill", k != EAGER, k.isdigit() and int(k)),
    )
    rungs = [k for k in keys if k != "prefill"]
    # Each solve: its column names, its rows, and their design matrix.
    solves = [("prefill", list(HISTORY if form == "history" else PREFILL))]
    if form == "shared" and rungs:
        solves.append(
            (None, ["context", "graph_padding"] + [f"step@{k}" for k in rungs])
        )
    else:
        solves += [(k, list(DECODE)) for k in rungs]
    for key, names in solves:
        members = [
            r for r in rows if r[1] == key or (key is None and r[1] != "prefill")
        ]
        if not members:
            law["unfitted"][key] = f"0 steps for {len(names)} coefficients"
            continue
        counts = np.array([r[2] for r in members], dtype=np.float64)
        x = counts
        if key is None:
            onehot = [[float(r[1] == k) for k in rungs] for r in members]
            x = np.column_stack([counts[:, 1:], onehot])
        solved = _solve(x, np.array([r[3] for r in members]), names)
        slow = ~solved["keep"] & (solved["residual"] < 0)
        once = solved["residual"] < 1 / WARM_FACTOR - 1
        warm = np.zeros(len(members), dtype=bool)
        for rec in range(len(records)):
            leading = True
            for j in (j for j, r in enumerate(members) if r[0] == rec):
                leading = leading and slow[j]
                if not (leading or slow[j] and once[j]):
                    continue
                warm[j] = True
                law["warm_steps"][rec].append(members[j][4])
                cut["warm"] += 1
                cut["warm_s"] += members[j][3]
        for k in rungs if key is None else [key]:
            mine = np.array([r[1] == k for r in members])
            coef = solved["coef"]
            if key is None:
                coef = (coef[2 + rungs.index(k)], *coef[:2])
            features = names if key else DECODE
            kept = mine & solved["keep"]
            try:
                if kept.sum() <= len(features):
                    raise ValueError(
                        f"{int(kept.sum())} steps for {len(features)} coefficients"
                    )
                hull = _hull(counts[kept][:, 1:], tuple(features[1:]))
            except (ValueError, QhullError) as e:
                law["unfitted"][k] = str(e).splitlines()[0]
                continue
            vector = {
                "coefficients": dict(zip(features, map(float, coef))),
                "hull": hull,
                "fit": {
                    "samples": int(kept.sum()),
                    "warm": int((mine & warm).sum()),
                    "dropped": int((mine & ~solved["keep"] & ~warm).sum()),
                    "condition": solved["condition"],
                    "zero": solved["zero"],
                },
            }
            if k == "prefill":
                law["prefill"] = vector
            else:
                law["decode"][k] = vector
    return law


def _band(key: str, counts) -> str:
    if key != "prefill":
        return f"decode rung {key}"
    lo = max([0] + [b for b in TOKEN_BANDS if b <= counts[1]])
    hi = min([b for b in TOKEN_BANDS if b > counts[1]], default="")
    return f"prefill tokens {lo}-{hi}"


def held_out(records, form: str = "law") -> dict:
    """Per band, the relative errors on each record of the law fitted on the others."""
    warm = fit(records, form)["warm_steps"]
    errors: dict[str, list[float]] = {}
    for i, record in enumerate(records):
        law = fit(records[:i] + records[i + 1 :], form)
        for key, counts, seconds, step in samples(record, form)[0]:
            if step in warm[i]:
                continue
            vector = law["decode"].get(key) if key != "prefill" else law.get(key)
            band = _band(key, counts)
            if vector is None:
                errors.setdefault(f"{band} (no vector)", []).append(float("nan"))
                continue
            price = np.dot(list(vector["coefficients"].values()), counts)
            errors.setdefault(band, []).append(float(price) / seconds - 1.0)
    return errors


def _table(errors: dict) -> str:
    lines = ["band | steps | p10 | p50 | p90"]
    for band in sorted(errors):
        e = np.array(errors[band])
        p = ["-"] * 3
        if not np.isnan(e).any():
            p = [f"{v:+.1%}" for v in np.percentile(e, (10, 50, 90))]
        lines.append(f"{band} | {len(e)} | " + " | ".join(p))
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m atom.compass.backends.coarse_fit",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("records", nargs="+", help="real step records, steps/dp0.jsonl")
    parser.add_argument(
        "--out", required=True, help="where the law is written, as JSON"
    )
    parser.add_argument("--form", choices=FORMS, default="law")
    args = parser.parse_args(argv)
    law = fit(args.records, args.form)
    Path(args.out).write_text(json.dumps(law) + "\n")
    print(f"cut: {json.dumps(law['cut'])}")
    for name, vector in [("prefill", law.get("prefill"))] + list(law["decode"].items()):
        if vector:
            print(
                f"{name}: {json.dumps(vector['coefficients'])} {json.dumps(vector['fit'])}"
            )
    if len(args.records) > 1:
        for form in FORMS:
            print(f"\nheld out, form {form}\n{_table(held_out(args.records, form))}")


if __name__ == "__main__":
    main()
