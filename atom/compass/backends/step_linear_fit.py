# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The step linear law fitted from real step records.

    python -m atom.compass.backends.step_linear_fit RECORD... --out LAW [--form FORM] [--dp N]

Each RECORD is a real run's step record (`steps/dp0.jsonl`, written with
`ATOM_COMPASS_PARITY_RECORD`). A step's label is its device busy time,
`d_busy_ns`, plus for a decode step the device work queued since the previous
forward's exit, `d_gap_ns`. A decode step whose device start precedes its host
launch end is launch-exposed, counted and not fitted; a prefill step is never
cut, so a host bubble inside it stays in its label. The exit of each wait for the
previous output's token copy (`t_recv_ns`) bounds the host clock's lead over
the device's (that forward had finished by then); the tightest of the last
`ANCHORS` bounds is the lead. The copy itself lands on its own stream after the
exit event the bound reads, by 14 to 29 us measured on one host (median 17 us),
so each bound carries that much more lead than the clocks actually have; on
this law's records the gap never moves a decode step across the
launch-exposed cut, so it is left in the bound rather than corrected by a
host-specific number with no record or law of its own to carry it. A record
without device windows is refused. A step mixing prefill and decode rows is
counted and not fitted.

With `--dp N` (EP off) the records come in runs of N, each run's `dp0.jsonl`
to `dp<N-1>.jsonl`, and the law is keyed `(N, 1)`. A step is its ranks' lines
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
after `MAD_PASSES` refits. Held-out errors are
printed at `--dp 1` only.

Records with a host key (`dp0.key.json`, from the real runner) also give the
host laws as a run file's `host` entry: `key`, and `laws[kind][segment]`,
seconds on `(1, tokens, seqs)`. `E` runs from the previous forward's exit,
cut as idle when the two batches share no request; `W_pre` to the launch, less
the staging gate's wait; `W_post` from the launch to the exit, less the token
copy wait, and a middle chunk has none. They take the step linear law's
solver, MAD drops and hulls; `seqs` equal to `tokens` on every sample gets
zero. Two keys refuse.

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

Forms (`FORMS`): `law`, the features `StepLinearLaw` always prices; `history`,
prefill plus `Sum (N_KV - N_Q)`, the KV a short chunk reads on a deep history;
`shared`, decode with the context and padding slopes shared by every rung and
an intercept per rung.

Outside `shared`, a decode rung may adopt a `rowmod(k)` column (`StepLinearLaw`),
read from its rows in scheduled order, by the rule `_rowmod` states; the
vector then names `k` and records, under `fit.rowmod`, the MAD without the
column, the mean MAD with each step's rows shuffled, and per held-out record
the p50 error without and with it.

With two or more records it also prints, per form, the error on each record of
a law fitted on the others, warm steps left out: per kind and rung, or prefill
token band, the p10, p50 and p90 of `predicted / measured - 1`; and the same
per host law.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np

from atom.compass.backends.shape import (
    BatchView,
    RequestShape,
    sum_context,
    sum_query_cached,
    sum_query_square,
    sum_tokens,
)
from atom.compass.backends.step_linear import (
    DECODE,
    EAGER,
    HISTORY,
    PREFILL,
    ROWMOD,
    rowmod,
)

FORMS = ("law", "history", "shared")
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
#: Token copy wait exits whose tightest bound is a step's host clock lead.
ANCHORS = 32
HOST_KINDS = ("decode", "prefill", "middle")
HOST_SEGMENTS = ("E", "W_pre", "W_launch", "W_post")
HOST_FEATURES = ("step", "tokens", "seqs")


def _rung_key(rung: int | None) -> str:
    return EAGER if rung is None else str(rung)


def _lines(record) -> list[dict]:
    return [json.loads(line) for line in Path(record).read_text().splitlines()]


def _launch_exposed(lines) -> set[int]:
    """The steps whose device start precedes their host launch end."""
    device, output_end, timed = 0, None, []
    for s in lines:
        if "d_busy_ns" not in s:
            continue
        start = device + s.get("d_gap_ns", 0)
        device = start + s["d_busy_ns"]
        bound = None
        if "t_recv_ns" in s and output_end is not None:
            bound = s["t_recv_ns"][1] - output_end
        if s["produces_output"]:
            output_end = device
        timed.append((s, start, bound))
    bounds = [b for _, _, b in timed if b is not None]
    if not bounds:
        raise ValueError(
            "no token copy wait exit (`t_recv_ns`) ties the record's host and device clocks"
        )
    # Steps before the first anchor read the first ones.
    first, recent, exposed = min(bounds[:ANCHORS]), deque(maxlen=ANCHORS), set()
    for s, start, bound in timed:
        if bound is not None:
            recent.append(bound)
        if start + min(recent, default=first) < s["t_launch_ns"][1]:
            exposed.add(s["step"])
    return exposed


def samples(record, form: str = "law", dp: int = 1) -> tuple[list[tuple], dict]:
    """One record's fittable steps as (vector, features, seconds, step, context), and
    what was cut.

    `vector` is `prefill` or a decode rung's key; features are in `StepLinearLaw`'s
    order, plus `history` for that form and, above `dp` 1, `moe_rows` on prefill;
    `context` is the decode rows' N_KV in scheduled order, empty for prefill.
    """
    lines = _lines(record)
    steps = [s for s in lines if not s["is_dummy_run"]]
    if not steps or "t_enter_ns" not in steps[0]:
        raise ValueError(
            f"{record} has no timed steps; a fit needs a real run's record"
        )
    if "d_busy_ns" not in steps[0]:
        raise ValueError(
            f"{record} has no device windows (`d_busy_ns`); a fit labels a step "
            "by its device time"
        )
    exposed = _launch_exposed(lines)
    cut = {"exposed": 0, "mixed": 0}
    rows = []
    for step in steps:
        view = BatchView(
            tuple(RequestShape(q, c, bool(d)) for q, c, d in step["rows"]),
            capture_rung=step["rung"],
        )
        if view.prefill and view.decode:
            cut["mixed"] += 1
        elif not view.prefill and step["step"] in exposed:
            cut["exposed"] += 1
        else:
            if view.prefill:
                r = view.prefill
                key = "prefill"
                counts = (1, sum_tokens(r), sum_query_square(r), sum_query_cached(r))
                if form == "history":
                    counts += (sum_context(r) - sum_tokens(r),)
                if dp > 1:
                    if step.get("moe_rows") is None:
                        raise ValueError(
                            f"{record} step {step['step']} has no moe_rows, which "
                            "a data-parallel law prices prefill from"
                        )
                    counts += (step["moe_rows"],)
            else:
                key = _rung_key(view.capture_rung)
                counts = (1, sum_context(view.decode), view.graph_padding)
            gap = 0 if view.prefill else step.get("d_gap_ns", 0)
            label = (gap + step["d_busy_ns"]) * 1e-9
            context = tuple(r.context_tokens for r in view.decode)
            rows.append((key, counts, label, step["step"], context))
    return rows, cut


def _shared(step, other) -> bool:
    """Whether two steps' batches hold a request in common."""
    return bool({r[0] for r in step["batch"]} & {r[0] for r in other["batch"]})


def host_samples(record) -> tuple[list[tuple], int]:
    """(kind, segment, (1, tokens, seqs), seconds) per host segment of each
    real forward, and how many `E` gaps were cut as idle."""
    rows, idle, prev = [], 0, None
    for s in _lines(record):
        if s["is_dummy_run"]:
            prev = s
            continue
        enter, (l0, l1) = s["t_enter_ns"], s["t_launch_ns"]
        g0, g1 = s.get("t_gate_ns", (l0, l0))
        r0, r1 = s.get("t_recv_ns", (0, 0))
        kind = "decode" if all(d for _, _, d in s["rows"]) else "prefill"
        kind = kind if s["produces_output"] else "middle"
        segments = {"W_pre": g0 - enter + l0 - g1, "W_launch": l1 - l0}
        if kind != "middle":
            segments["W_post"] = s["t_exit_ns"] - l1 - (r1 - r0)
        if prev is not None and _shared(s, prev):
            segments["E"] = enter - prev["t_exit_ns"]
        elif prev is not None:
            idle += 1
        x = (1, sum(r[1] for r in s["batch"]), len(s["batch"]))
        rows += [(kind, k, x, v * 1e-9) for k, v in segments.items()]
        prev = s
    return rows, idle


def _key(record):
    path = Path(record).with_suffix(".key.json")
    return json.loads(path.read_text()) if path.exists() else None


def fit_host(records) -> tuple[dict, dict]:
    """The host laws over `records`, as a run file's `host` entry, and their report."""
    from scipy.spatial import QhullError

    keys = sorted({json.dumps(_key(r), sort_keys=True) for r in records})
    if len(keys) != 1:
        raise ValueError(
            f"the records carry {len(keys)} host keys ({' / '.join(keys)}); "
            "host laws are fitted under one"
        )
    got = [host_samples(r) for r in records]
    rows = [row for r, _ in got for row in r]
    host = {"key": json.loads(keys[0]), "laws": {}}
    report = {"cut": {"idle": sum(n for _, n in got)}, "unfitted": {}}
    for kind in HOST_KINDS:
        for segment in HOST_SEGMENTS[: 3 if kind == "middle" else 4]:
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


def _rowmod(members, rung: int, base: dict) -> tuple | None:
    """The `rowmod(k)` column decode rung `rung` adopts, or None.

    `members` are the rung's `fit` rows, `base` their solve without the column.
    `k` runs over the powers of two up to the rung. A `k` qualifies when its
    solve cuts `base`'s scaled MAD by `ROWMOD_GAIN`, the mean MAD of the same
    solve over `ROWMOD_SHUFFLES` seeded shuffles of each step's rows does not,
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
    rng = np.random.default_rng(0)
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
    """One run's steps, each its ranks' `samples` rows labelled with the smallest
    rank label; what was cut; and each step's rank label spread."""
    got = [samples(record, form, len(run)) for record in run]
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


def fit(records, form: str = "law", dp: int = 1) -> dict:
    """The law over `records`, runs of `dp` ranks each, as the mapping `StepLinearLaw` reads."""
    from scipy.spatial import QhullError

    if form not in FORMS:
        raise ValueError(f"form {form!r} is not one of {FORMS}")
    if len(records) % dp:
        raise ValueError(f"{len(records)} records are not runs of {dp} ranks")
    runs = [records[i : i + dp] for i in range(0, len(records), dp)]
    groups, cut, spread = [], {"warm": 0, "warm_s": 0.0}, []
    for i, run in enumerate(runs):
        got, c, s = _groups(run, form)
        groups += [[(i, *r) for r in g] for g in got]
        spread += s
        for k, v in c.items():
            cut[k] = cut.get(k, 0) + v
    prefill = list(HISTORY if form == "history" else PREFILL) + ["moe_rows"] * (dp > 1)
    rows, alternation = _slowest(groups, prefill)
    digests = [
        f"{Path(r).name}:{hashlib.sha256(Path(r).read_bytes()).hexdigest()[:12]}"
        for r in records
    ]
    law = {
        "provenance": f"coarse_fit {form} on {len(records)} real records "
        f"({', '.join(digests)})",
        "form": form,
        "cut": cut,
        "key": [dp, 1],
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
            chosen = _rowmod(members, int(key), solved)
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
    if any(_key(r) for r in records):
        law["host"], law["host_fit"] = fit_host(records)
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
        for key, counts, seconds, step, context in samples(record, form)[0]:
            if step in warm[i]:
                continue
            vector = law["decode"].get(key) if key != "prefill" else law.get(key)
            band = _band(key, counts)
            if vector is None:
                errors.setdefault(f"{band} (no vector)", []).append(float("nan"))
                continue
            if "rowmod" in vector:
                counts += (rowmod(context, vector["rowmod"]),)
            price = np.dot(list(vector["coefficients"].values()), counts)
            errors.setdefault(band, []).append(float(price) / seconds - 1.0)
    return errors


def host_held_out(records) -> dict:
    """Per host law, the relative errors on each record of the laws fitted on the others."""
    errors: dict[str, list[float]] = {}
    for i, record in enumerate(records):
        laws = fit_host(records[:i] + records[i + 1 :])[0]["laws"]
        for kind, segment, x, seconds in host_samples(record)[0]:
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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m atom.compass.backends.step_linear_fit",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("records", nargs="+", help="real step records, steps/dp0.jsonl")
    parser.add_argument(
        "--out", required=True, help="where the law is written, as JSON"
    )
    parser.add_argument("--form", choices=FORMS, default="law")
    parser.add_argument(
        "--dp", type=int, default=1, help="ranks per run, the records in rank order"
    )
    args = parser.parse_args(argv)
    law = fit(args.records, args.form, args.dp)
    Path(args.out).write_text(json.dumps(law) + "\n")
    print(f"cut: {json.dumps(law['cut'])}")
    for name, vector in [("prefill", law.get("prefill"))] + list(law["decode"].items()):
        if vector:
            print(
                f"{name}: {json.dumps(vector['coefficients'])} {json.dumps(vector['fit'])}"
            )
    if "host" in law:
        print(f"host: {json.dumps(law['host']['laws'])} {json.dumps(law['host_fit'])}")
    if len(args.records) > 1 and args.dp == 1:
        for form in FORMS:
            print(f"\nheld out, form {form}\n{_table(held_out(args.records, form))}")
        if "host" in law:
            print(f"\nheld out, host laws\n{_table(host_held_out(args.records))}")


if __name__ == "__main__":
    main()
