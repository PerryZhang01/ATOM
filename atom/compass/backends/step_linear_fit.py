# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The step linear law fitted from real step records.

    python -m atom.compass.backends.step_linear_fit RECORD... --out LAW --machine-spec DIGEST [--form FORM] [--engine FILE]

Each RECORD is a real run's step record (`steps/dp0.jsonl`, written with
`ATOM_COMPASS_PARITY_RECORD`), labelled by `step_labels`, whose docstring
states the labels; the command writes each record's per-step table beside it.
A step whose device label is cut is counted under its reason and not fitted.
`fit` and `fit_host` take the labelled samples, grouped into runs by their
provenance's `run`, and each returns its laws and a report. `fit`'s report
holds the runs it was fitted on, each vector's fit figures and hull, and, once
`mark_held_out` has run, the held-out errors per run; its `seed` fixes the
`rowmod` shuffles, so the same samples and arguments give the same bytes.

The law states the records' `engine_config`, read from the `.engine.json`
beside each (`engine_config`) or from `--engine FILE` in their place, and a run
under another engine config refuses it. At its `data_parallel_size` N above 1
(EP off) the records come in runs of
N, each run's `dp0.jsonl` to `dp<N-1>.jsonl`. A step is its ranks' lines
joined on step index, labelled with the smallest rank label (the others wait
for the last rank to enter), and a prefill row adds the line's `moe_rows`. A
step that some rank cut, or whose ranks' rows fall to different vectors, is
counted under `ranks` and not fitted. The law reports each step's rank label
spread, `max / min - 1`, as its p50 and p90 under `spread`. The group's step
is the max over its ranks, so each vector is fitted by alternating max-affine
regression: from the fit on every rank's row, take each step's row the law
prices highest and refit on those rows, until no step changes rank. Each
vector reports, under `fit.all_rows`, the centre and scaled MAD of the fit on
every rank's row, each step priced as its max over ranks, and under
`fit.passes` the refits the alternation took, null when the picks still moved
after `MAD_PASSES` refits. A held-out step is
priced and banded as its costliest rank.

Records with a host key (`dp0.key.json`, from the real runner) also give the
host laws as a run file's `host` entry: `key`, and `laws[kind][segment]`,
seconds on `(1, tokens, seqs)`, fitted on the samples' host labels. They take
the step linear law's solver, MAD drops and hulls; `seqs` equal to `tokens` on
every sample gets zero. Two keys refuse. The law and the host key state the
`--machine-spec` digest, which a run's spec must match, and the host key's `engine` is the
law's `engine_config`. A segment outside `HOST_SEGMENTS` is
fitted for a kind whose samples carry it: `W_inputs`, on an MTP record, and
`W_recv`, on a later pipeline stage's.

Each equation is divided by its own label and the coefficients are bound to be
non-negative. Steps whose relative residual is more than `MAD_K` scaled MADs
from the median are dropped and the rest refitted, until the kept set settles.
The dropped steps that lead a record's steps of one vector, each slower than
the law, are its warm steps (shape compiles and first touches), and so is a
dropped step anywhere that ran more than `WARM_FACTOR` times its price, such as
the first short chunk on a deep history: counted with their seconds apart from
the dropped ones, never charged. A feature with no variance gets coefficient
zero and is named. Each vector reports its samples,
warm and dropped counts, the centre and scaled MAD of its relative residuals,
the condition number of its column-scaled design and
the convex hull of its fitted steps, which `StepLinearLaw` reads. A vector with
no more fitted steps than coefficients, or whose steps span no hull, is left
out and named under `unfitted`; `StepLinearBackend` refuses its steps.

On an MTP record a step also carries a draft label (`step_labels`). `fit` fits
the steps that carry one on their draft labels the same way, and writes that law
under the law's `draft`; it holds no `held_out`.

Forms (`FORMS`): `base`, the features `StepLinearLaw` always prices; `history`,
prefill plus `Sum (N_KV - N_Q)`, the KV a short chunk reads on a deep history;
`shared`, decode with the context and padding slopes shared by every rung and
an intercept per rung.

Outside `shared`, a decode rung may adopt a `rowmod(k)` column (`StepLinearLaw`),
read from its rows in scheduled order, by the rule `_rowmod` states; the
vector then names `k` and records, under `fit.rowmod`, the MAD without the
column, the mean MAD with each step's rows shuffled, and per held-out record
the p50 error without and with it.

With two or more runs it also prints, per form, the error on each run of a
law fitted on the others, warm steps left out: per kind and rung, or prefill
token band, the p10, p50 and p90 of `predicted / measured - 1`; and the same
per host law. The law's own form's errors are written into it: under each
vector's `held_out`, the convex hull of every rank's steps they were measured on and
their percentiles per band, which `StepLinearLaw` reads to mark a term
validated. A vector whose held-out steps span no hull gets none.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import numpy as np

from atom.compass.backends import step_labels
from atom.compass.backends.shape import (
    sum_context,
    sum_query_cached,
    sum_query_square,
    sum_tokens,
)
from atom.compass.backends.step_grid_plan import keyed_of
from atom.compass.backends.step_labels import CUTS, HOST_SEGMENTS, host_kind
from atom.compass.backends.step_linear import (
    DECODE,
    EAGER,
    HISTORY,
    PREFILL,
    ROWMOD,
    rowmod,
)

FORMS = ("base", "history", "shared")
MAD_K = 3.5
#: A relative residual this small is never an outlier, however tight the rest.
MAD_FLOOR = 0.01
MAD_PASSES = 20
#: A step that ran this many times its price is a one-off first touch, not scatter.
WARM_FACTOR = 5.0
TOKEN_BANDS = (512, 2048, 8192)
#: Steps at its rung a record needs before a `rowmod` choice holds it out.
ROWMOD_CELL_STEPS = 100
#: The fraction of a rung's scaled MAD a `rowmod` column must cut: a smaller cut is
#: within the sampling error of a MAD over a few hundred steps.
ROWMOD_GAIN = 0.1
#: Seeded shuffles of each step's rows whose mean MAD the `rowmod` column must not beat.
ROWMOD_SHUFFLES = 8
HOST_KINDS = ("decode", "prefill", "middle")
HOST_FEATURES = ("step", "tokens", "seqs")


def _rung_key(rung: int | None) -> str:
    return EAGER if rung is None else str(rung)


def _features(samples, form: str, dp: int) -> tuple[list[tuple], dict]:
    """One rank's fittable samples as (vector, features, seconds, step, context),
    and how many were cut, per reason.

    `vector` is `prefill` or a decode rung's key; features are in `StepLinearLaw`'s
    order, plus `history` for that form and, above `dp` 1, `moe_rows` on prefill;
    `context` is the decode rows' N_KV in scheduled order, empty for prefill.
    """
    cut, rows = dict.fromkeys(CUTS, 0), []
    for s in samples:
        view = s.view
        if s.cut:
            cut[s.cut] += 1
        else:
            if view.prefill:
                r = view.prefill
                key = "prefill"
                counts = (1, sum_tokens(r), sum_query_square(r), sum_query_cached(r))
                if form == "history":
                    counts += (sum_context(r) - sum_tokens(r),)
                if dp > 1:
                    if view.moe_rows is None:
                        raise ValueError(
                            f"{s.provenance['run']} step {s.step} has no moe_rows, which "
                            "a data-parallel law prices prefill from"
                        )
                    counts += (view.moe_rows,)
            else:
                key = _rung_key(view.capture_rung)
                counts = (1, sum_context(view.decode), view.graph_padding)
            context = tuple(r.context_tokens for r in view.decode)
            rows.append((key, counts, s.device_s, s.step, context))
    return rows, cut


def _host_rows(samples) -> list[tuple]:
    """(kind, segment, (1, tokens, seqs), seconds) per host label of each sample."""
    rows = []
    for s in samples:
        x = (1, sum_tokens(s.view.requests), len(s.view.requests))
        rows += [(host_kind(s.view), k, x, v) for k, v in s.host.items()]
    return rows


def label_records(records, dp: int = 1) -> tuple[list, dict]:
    """`records`, runs of `dp` ranks in rank order, as `step_labels` samples, each
    run named by its records' digests; and each record's table rows."""
    samples, tables = [], {}
    for i in range(0, len(records), dp):
        run = records[i : i + dp]
        name = ", ".join(map(step_labels.digest, run))
        for record in run:
            got, tables[record] = step_labels.label(record, name)
            samples += got
    return samples, tables


def _runs(samples, dp: int) -> dict[str, list[list]]:
    """`samples` per run, in the order runs first appear, as each rank's samples."""
    runs: dict[str, dict[int, list]] = {}
    for s in samples:
        runs.setdefault(s.provenance["run"], {}).setdefault(s.rank, []).append(s)
    if any(len(ranks) != dp for ranks in runs.values()):
        n = sum(map(len, runs.values()))
        raise ValueError(f"{n} records are not runs of {dp} ranks")
    return {run: [ranks[r] for r in sorted(ranks)] for run, ranks in runs.items()}


def fit_host(samples) -> tuple[dict, dict]:
    """The host laws over `samples`, as a run file's `host` entry, and their report."""
    from scipy.spatial import QhullError

    keys = sorted({json.dumps(s.provenance["config"], sort_keys=True) for s in samples})
    if len(keys) != 1:
        raise ValueError(
            f"the records carry {len(keys)} host keys ({' / '.join(keys)}); "
            "host laws are fitted under one"
        )
    rows = _host_rows(samples)
    host = {"key": json.loads(keys[0]), "laws": {}}
    report = {"cut": {"idle": sum(s.idle for s in samples)}, "unfitted": {}}
    for kind in HOST_KINDS:
        extra = sorted({r[1] for r in rows if r[0] == kind} - set(HOST_SEGMENTS))
        for segment in HOST_SEGMENTS[: 3 if kind == "middle" else 4] + tuple(extra):
            members = [r for r in rows if r[:2] == (kind, segment) and r[3] > 0]
            x = np.array([r[2] for r in members], dtype=np.float64).reshape(-1, 3)
            width = 2 if np.array_equal(x[:, 1], x[:, 2]) else 3
            if width == 2:
                x[:, 2] = 0.0
            try:
                if len(members) <= len(HOST_FEATURES):
                    raise ValueError(f"{len(members)} forwards for 3 coefficients")
                s = _solve(x, np.array([r[3] for r in members]), list(HOST_FEATURES))
                hull = _hull(x[s["keep"]][:, 1:width], HOST_FEATURES[1:width])
            except (ValueError, QhullError) as e:
                report["unfitted"][f"{kind}.{segment}"] = str(e).splitlines()[0]
                continue
            host["laws"].setdefault(kind, {})[segment] = s["coef"].tolist()
            report[f"{kind}.{segment}"] = {
                "hull": hull,
                "samples": int(s["keep"].sum()),
                "dropped": int((~s["keep"]).sum()),
                **{k: s[k] for k in ("condition", "centre", "mad", "zero")},
            }
    return host, report


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
        # A column with no non-zero kept row (a rung this pass dropped every
        # step of) sits out of this pass's solve, at coefficient 0.
        solved = norms > 0
        cols, norms = cols[:, solved], norms[solved]
        coef = np.zeros(x.shape[1])
        coef[np.array(vary)[solved]] = nnls(cols / norms, np.ones(len(cols)))[0] / norms
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
        "centre": float(centre),
        "mad": float(mad),
        "condition": float(np.linalg.cond(cols / norms)),
        "zero": [names[j] for j in range(x.shape[1]) if j not in vary],
    }


def _rowmod(members, rung: int, base: dict, seed: int) -> tuple | None:
    """The `rowmod(k)` column decode rung `rung` adopts, or None.

    `members` are the rung's `fit` rows, `base` their solve without the column.
    `k` runs over the powers of two up to the rung. A `k` qualifies when its
    solve cuts `base`'s scaled MAD by `ROWMOD_GAIN`, the mean MAD of the same
    solve over `ROWMOD_SHUFFLES` shuffles of each step's rows, by `seed`, does not,
    and no record (a cell) priced by a solve on the others has a p50 error
    further from zero than without the column. Every cell
    needs `ROWMOD_CELL_STEPS` steps at the rung, and a rung needs two cells.
    The qualifying `k` of smallest MAD is adopted: returned with its column,
    its solve and the evidence the law records.
    """
    cell = np.array([r[0] for r in members])
    cells = sorted(set(cell.tolist()))
    if len(cells) < 2 or min((cell == c).sum() for c in cells) < ROWMOD_CELL_STEPS:
        return None
    x = np.array([r[2] for r in members], dtype=np.float64)
    y = np.array([r[3] for r in members])
    rng = np.random.default_rng(seed)
    shuffles = [
        [rng.permutation(r[5]) for r in members] for _ in range(ROWMOD_SHUFFLES)
    ]

    def mad(rows, k):
        column = [rowmod(r, k) for r in rows]
        return _solve(np.column_stack([x, column]), y, list(ROWMOD))["mad"]

    def p50(design):
        names = list(ROWMOD[: design.shape[1]])
        out = []
        for c in cells:
            other = cell != c
            coef = _solve(design[other], y[other], names)["coef"]
            out.append(float(np.median(design[~other] @ coef / y[~other] - 1.0)))
        return out

    without, best, k = p50(x), None, 2
    target = (1.0 - ROWMOD_GAIN) * base["mad"]
    while k <= rung:
        column = np.array([rowmod(r[5], k) for r in members], dtype=np.float64)
        design = np.column_stack([x, column])
        solved = _solve(design, y, list(ROWMOD))
        if solved["mad"] < min(target, best[2]["mad"] if best else target):
            shuffled_mad = float(np.mean([mad(s, k) for s in shuffles]))
            held = p50(design)
            if shuffled_mad >= target and all(
                abs(w) <= abs(o) for w, o in zip(held, without)
            ):
                evidence = {
                    "mad_without": base["mad"],
                    "shuffled_mad": shuffled_mad,
                    "held_out_p50": {
                        str(c): [o, w] for c, o, w in zip(cells, without, held)
                    },
                }
                best = (k, column, solved, evidence)
        k *= 2
    return best


def _hull(points: np.ndarray, names: tuple[str, ...]) -> dict:
    """`StepLinearLaw`'s hull over the non-intercept features of the fitted steps."""
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


def _groups(run, form: str) -> tuple[list[list[tuple]], dict, list[float]]:
    """One run's steps, each its ranks' `_features` rows labelled with the smallest
    rank label; what was cut; and each step's rank label spread."""
    got = [_features(rank, form, len(run)) for rank in run]
    cut = {k: sum(c[k] for _, c in got) for k in got[0][1]}
    if len(run) == 1:
        return [[r] for r in got[0][0]], cut, []
    ranks = [{r[3]: r for r in rows} for rows, _ in got]
    groups, spread, cut["ranks"] = [], [], 0
    for step in sorted(set().union(*ranks)):
        group = [rank.get(step) for rank in ranks]
        if None in group or len({r[0] for r in group}) > 1:
            cut["ranks"] += 1
            continue
        labels = [r[2] for r in group]
        spread.append(max(labels) / min(labels) - 1.0)
        groups.append([(*r[:2], min(labels), *r[3:]) for r in group])
    return groups, cut, spread


def _slowest(groups, prefill: list[str]) -> tuple[list[tuple], dict]:
    """Each group's row of highest price, by alternating max-affine regression per
    vector, and per vector the error of the fit on every rank's row."""
    rows, report = [g[0] for g in groups], {}
    for key in sorted({g[0][1] for g in groups if len(g) > 1}):
        names = prefill if key == "prefill" else list(DECODE)
        at = [j for j, g in enumerate(groups) if g[0][1] == key]
        x = np.array([[r[2] for r in groups[j]] for j in at], dtype=np.float64)
        y = np.array([groups[j][0][3] for j in at])
        flat = x.reshape(-1, x.shape[2])
        coef = _solve(flat, np.repeat(y, x.shape[1]), names)["coef"]
        error = (x @ coef).max(axis=1) / y - 1.0
        centre = float(np.median(error))
        mad = float(1.4826 * np.median(np.abs(error - centre)))
        pick = None
        for passes in range(MAD_PASSES + 1):
            now = np.argmax(x @ coef, axis=1)
            if pick is not None and (now == pick).all():
                break
            if passes == MAD_PASSES:
                passes = None
                break
            pick = now
            coef = _solve(x[np.arange(len(x)), pick], y, names)["coef"]
        report[key] = {"all_rows": {"centre": centre, "mad": mad}, "passes": passes}
        for j, p in zip(at, pick):
            rows[j] = groups[j][p]
    return rows, report


def _options(options) -> tuple[str, int]:
    options = {"form": "base", "dp": 1, **(options or {})}
    if options["form"] not in FORMS:
        raise ValueError(f"form {options['form']!r} is not one of {FORMS}")
    return options["form"], options["dp"]


def fit(samples, options=None, seed: int = 0) -> tuple[dict, dict]:
    """The law over `samples`, as the mapping `StepLinearLaw` reads, and its report.

    `options` takes `form` (`base` by default) and `dp`, the ranks each run
    holds (1 by default); `seed` fixes the `rowmod` shuffles.
    """
    from scipy.spatial import QhullError

    form, dp = _options(options)
    runs = _runs(samples, dp)
    groups, cut, spread = [], {"warm": 0, "warm_s": 0.0}, []
    for i, run in enumerate(runs.values()):
        got, c, s = _groups(run, form)
        groups += [[(i, *r) for r in g] for g in got]
        spread += s
        for k, v in c.items():
            cut[k] = cut.get(k, 0) + v
    prefill = list(HISTORY if form == "history" else PREFILL) + ["moe_rows"] * (dp > 1)
    rows, alternation = _slowest(groups, prefill)
    law = {
        "provenance": f"step_linear_fit {form} on {len(runs) * dp} real records "
        f"({', '.join(runs)})",
        "form": form,
        "cut": cut,
        "warm_steps": [[] for _ in runs],
        "unfitted": {},
        "decode": {},
    }
    if spread:
        law["spread"] = np.percentile(spread, (50, 90)).tolist()
    keys = sorted(
        {r[1] for r in rows},
        key=lambda k: (k != "prefill", k != EAGER, k.isdigit() and int(k)),
    )
    rungs = [k for k in keys if k != "prefill"]
    # Each solve: its column names, its rows, and their design matrix.
    solves = [("prefill", prefill)]
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
        rowmod_k = None
        if key and key.isdigit():
            chosen = _rowmod(members, int(key), solved, seed)
            if chosen:
                rowmod_k, column, solved, evidence = chosen
                counts = np.column_stack([counts, column])
                names = list(ROWMOD)
        slow = ~solved["keep"] & (solved["residual"] < 0)
        once = solved["residual"] < 1 / WARM_FACTOR - 1
        warm = np.zeros(len(members), dtype=bool)
        for rec in range(len(runs)):
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
                    "centre": solved["centre"],
                    "mad": solved["mad"],
                    "zero": solved["zero"],
                },
            }
            if k in alternation:
                vector["fit"].update(alternation[k])
            if rowmod_k:
                vector["rowmod"] = rowmod_k
                vector["fit"]["rowmod"] = evidence
            if k == "prefill":
                law["prefill"] = vector
            else:
                law["decode"][k] = vector
    vectors = [("prefill", law.get("prefill")), *law["decode"].items()]
    report = {
        "provenance": [
            {k: v for k, v in run[0][0].provenance.items() if k != "requests"}
            for run in runs.values()
        ],
        "vectors": {k: {**v["fit"], "hull": v["hull"]} for k, v in vectors if v},
        "held_out": {},
    }
    drafts = [
        dataclasses.replace(s, device_s=s.draft_s, draft_s=None)
        for s in samples
        if s.draft_s is not None
    ]
    if drafts:
        law["draft"] = fit(drafts, options, seed)[0]
    return law, report


def token_band(tokens: int) -> str:
    """The band of `TOKEN_BANDS` `tokens` falls in: `lo-hi`, or `lo+` past the last."""
    lo = max([0] + [b for b in TOKEN_BANDS if b <= tokens])
    hi = min([b for b in TOKEN_BANDS if b > tokens], default=None)
    return f"{lo}+" if hi is None else f"{lo}-{hi}"


def _band(key: str, counts) -> str:
    if key != "prefill":
        return f"decode rung {key}"
    return f"prefill tokens {token_band(counts[1])}"


def _held_out(samples, options, seed, warm=None):
    """Each step of each run, warm steps left out, priced as its costliest rank
    by the law fitted on the other runs: (run, vector, band, each rank's (counts,
    context), relative error), the error nan where that law has no vector."""
    form, dp = _options(options)
    runs = _runs(samples, dp)
    if warm is None:
        warm = fit(samples, options, seed)[0]["warm_steps"]
    for i, (run, ranks) in enumerate(runs.items()):
        others = [s for s in samples if s.provenance["run"] != run]
        law = fit(others, options, seed)[0]
        for group in _groups(ranks, form)[0]:
            key, _, seconds, step, _ = group[0]
            if step in warm[i]:
                continue
            vector = law["decode"].get(key) if key != "prefill" else law.get(key)
            mine = [(r[1], r[4]) for r in group]
            price, error = [c[1] for c, _ in mine], float("nan")
            if vector is not None:
                c = list(vector["coefficients"].values())
                price = [
                    float(np.dot(c, x + _rowmod_count(vector, n))) for x, n in mine
                ]
                error = max(price) / seconds - 1.0
            top = mine[int(np.argmax(price))][0]
            yield run, key, _band(key, top), mine, error


def _rowmod_count(vector: dict, context) -> tuple:
    return (rowmod(context, vector["rowmod"]),) if "rowmod" in vector else ()


def held_out(samples, options=None, seed: int = 0) -> dict:
    """Per band, the relative errors on each run of the law fitted on the others."""
    errors: dict[str, list[float]] = {}
    for _, _, band, _, error in _held_out(samples, options, seed):
        band = band if not math.isnan(error) else f"{band} (no vector)"
        errors.setdefault(band, []).append(error)
    return errors


def _bands(rows) -> dict:
    """Per band of `_held_out`'s rows, their steps and error percentiles."""
    bands = {}
    for band in sorted({r[2] for r in rows}):
        e = [r[4] for r in rows if r[2] == band]
        p10, p50, p90 = np.percentile(e, (10, 50, 90)).tolist()
        bands[band] = {"steps": len(e), "p10": p10, "p50": p50, "p90": p90}
    return bands


def mark_held_out(law: dict, report: dict, samples, options=None, seed: int = 0):
    """Write the held-out errors: under each of `law`'s vectors, as `held_out`,
    the hull of every rank's step `_held_out` priced, their errors per band; under
    `report`'s `held_out`, each run's errors per band."""
    from scipy.spatial import QhullError

    rows = [
        r
        for r in _held_out(samples, options, seed, law["warm_steps"])
        if not math.isnan(r[4])
    ]
    for run in dict.fromkeys(r[0] for r in rows):
        report["held_out"][run] = _bands([r for r in rows if r[0] == run])
    for key, vector in [("prefill", law.get("prefill")), *law["decode"].items()]:
        mine = [r for r in rows if r[1] == key]
        if vector is None or not mine:
            continue
        points = [c + _rowmod_count(vector, x) for r in mine for c, x in r[3]]
        try:
            hull = _hull(
                np.array(points, dtype=np.float64)[:, 1:],
                tuple(vector["coefficients"])[1:],
            )
        except (ValueError, QhullError):
            continue
        vector["held_out"] = {"hull": hull, "bands": _bands(mine)}


def host_held_out(samples) -> dict:
    """Per host law, the relative errors on each run of the laws fitted on the others."""
    errors: dict[str, list[float]] = {}
    for run in dict.fromkeys(s.provenance["run"] for s in samples):
        others = [s for s in samples if s.provenance["run"] != run]
        laws = fit_host(others)[0]["laws"]
        mine = [s for s in samples if s.provenance["run"] == run]
        for kind, segment, x, seconds in _host_rows(mine):
            c = laws.get(kind, {}).get(segment)
            if seconds > 0:
                errors.setdefault(
                    f"host {kind} {segment}" + ("" if c else " (no law)"), []
                ).append(float(np.dot(c, x)) / seconds - 1.0 if c else float("nan"))
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


def engine_config(records) -> dict:
    """The engine config `records` ran under: `keyed_of` the `.engine.json`
    beside each. A record without one, or records under two, refuse, naming
    the fields that differ."""
    held = []
    for record in records:
        engine = Path(record).with_suffix(".engine.json")
        if not engine.exists():
            raise ValueError(
                f"{record} has no {engine.name}, the engine config it ran under"
            )
        held.append(keyed_of(json.loads(engine.read_text())))
    names = sorted({k for e in held for k in e})
    if differ := [k for k in names if any(e.get(k) != held[0].get(k) for e in held)]:
        raise ValueError(
            f"the records ran under two engine configs, differing in {differ}"
        )
    return held[0]


def join_width(engine: dict) -> int:
    """The ranks a run's records join at: `engine`'s `data_parallel_size`."""
    if "data_parallel_size" not in engine:
        raise ValueError(
            "the records' engine config states no data_parallel_size, the ranks "
            "a step joins; it predates the field"
        )
    return engine["data_parallel_size"]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m atom.compass.backends.step_linear_fit",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("records", nargs="+", help="real step records, steps/dp0.jsonl")
    parser.add_argument(
        "--out", required=True, help="where the law is written, as JSON"
    )
    parser.add_argument("--form", choices=FORMS, default="base")
    parser.add_argument(
        "--machine-spec",
        required=True,
        help="the machine_spec_digest of the spec of the platform the records ran "
        "on, which keys the law and its host laws",
    )
    parser.add_argument(
        "--engine",
        help="the engine file the records ran under, stated in place of the "
        ".engine.json beside each; refused where one contradicts it",
    )
    args = parser.parse_args(argv)
    if args.engine:
        engine = keyed_of(json.loads(Path(args.engine).read_text()))
        for own in (Path(r).with_suffix(".engine.json") for r in args.records):
            held = keyed_of(json.loads(own.read_text())) if own.exists() else {}
            if differ := [k for k, v in held.items() if engine.get(k, v) != v]:
                raise ValueError(f"{own} contradicts {args.engine} in {differ}")
    else:
        engine = engine_config(args.records)
    dp = join_width(engine)
    options = {"form": args.form, "dp": dp}
    samples, tables = label_records(args.records, dp)
    for record, table in tables.items():
        step_labels.write_table(record, table)
    law, report = fit(samples, options)
    if any(s.provenance["config"] for s in samples):
        law["host"], law["host_fit"] = fit_host(samples)
    law["machine_spec"] = args.machine_spec
    law["engine_config"] = engine
    if "host" in law:
        law["host"]["key"] |= {"machine_spec": args.machine_spec, "engine": engine}
    if len(args.records) > dp:
        mark_held_out(law, report, samples, options)
    Path(args.out).write_text(json.dumps(law) + "\n")
    print(f"cut: {json.dumps(law['cut'])}")
    for name, vector in [("prefill", law.get("prefill"))] + list(law["decode"].items()):
        if vector:
            print(
                f"{name}: {json.dumps(vector['coefficients'])} {json.dumps(vector['fit'])}"
            )
    if "host" in law:
        print(f"host: {json.dumps(law['host']['laws'])} {json.dumps(law['host_fit'])}")
    if len(args.records) > dp:
        for form in FORMS:
            errors = held_out(samples, {"form": form, "dp": dp})
            print(f"\nheld out, form {form}\n{_table(errors)}")
        if "host" in law:
            print(f"\nheld out, host laws\n{_table(host_held_out(samples))}")


if __name__ == "__main__":
    main()
