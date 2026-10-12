# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Tier a's law fitted from real step records.

    python -m atom.compass.backends.coarse_fit RECORD... --out LAW [--form FORM]

Each RECORD is a real run's step record (`steps/dp0.jsonl`, written with
`ATOM_COMPASS_PARITY_RECORD`). A step's label is its device busy time,
`d_busy_ns`, plus for a decode step the device work queued since the previous
forward's exit, `d_gap_ns`. A decode step whose device start precedes its host
launch end is launch-exposed, counted and not fitted; a prefill step is never
cut, so a host bubble inside it stays in its label. The exit of each wait for the
previous output's token copy (`t_recv_ns`) bounds the host clock's lead over
the device's (that forward had finished by then); the tightest of the last
`ANCHORS` bounds is the lead. A record without device windows labels a step by
its period, `t_enter_ns` to the next step's, cutting a period that may hold an
idle gap. A step mixing prefill and decode rows is counted and not fitted.

Records with a host key (`dp0.key.json`, from the real runner) also give the
host laws as a run file's `host` entry: `key`, and `laws[kind][segment]`,
seconds on `(1, tokens, seqs)`. `E` runs from the previous forward's exit,
cut as idle when the two batches share no request; `W_pre` to the launch, less
the staging gate's wait; `W_post` from the launch to the exit, less the token
copy wait, and a middle chunk has none. They take tier a's solver, MAD drops
and hulls; `seqs` equal to `tokens` on every sample gets zero. Two keys refuse.

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
the convex hull of its fitted steps, which `CoarseLaw` reads. A vector with
no more fitted steps than coefficients, or whose steps span no hull, is left
out and named under `unfitted`; `CoarseBackend` refuses its steps.

Forms (`FORMS`): `law`, the features `CoarseLaw` always prices; `history`,
prefill plus `Sum (N_KV - N_Q)`, the KV a short chunk reads on a deep history;
`shared`, decode with the context and padding slopes shared by every rung and
an intercept per rung.

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


def samples(record, form: str = "law") -> tuple[list[tuple], dict]:
    """One record's fittable steps as (vector, features, seconds, step), and what was cut.

    `vector` is `prefill` or a decode rung's key; features are in `CoarseLaw`'s
    order, plus `history` for that form.
    """
    lines = _lines(record)
    steps = [s for s in lines if not s["is_dummy_run"]]
    if not steps or "t_enter_ns" not in steps[0]:
        raise ValueError(
            f"{record} has no timed steps; a fit needs a real run's record"
        )
    busy = "d_busy_ns" in steps[0]
    if busy:
        exposed = _launch_exposed(lines)
        cut = {"exposed": 0, "mixed": 0}
        pairs = zip(steps, steps[1:] + [None])
    else:
        exposed, cut, pairs = set(), {"idle": 0, "mixed": 0, "last": 1}, pairwise(steps)
    rows = []
    for step, nxt in pairs:
        idle = not busy and not _shared(step, nxt)
        view = BatchView(
            tuple(RequestShape(q, c, bool(d)) for q, c, d in step["rows"]),
            capture_rung=step["rung"],
        )
        if idle:
            cut["idle"] += 1
        elif view.prefill and view.decode:
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
            else:
                key = _rung_key(view.capture_rung)
                counts = (1, sum_context(view.decode), view.graph_padding)
            if busy:
                gap = 0 if view.prefill else step.get("d_gap_ns", 0)
                label = (gap + step["d_busy_ns"]) * 1e-9
            else:
                label = (nxt["t_enter_ns"] - step["t_enter_ns"]) * 1e-9
            rows.append((key, counts, label, step["step"]))
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
        "centre": float(centre),
        "mad": float(mad),
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
    rows, cut, labels = [], {"warm": 0, "warm_s": 0.0}, set()
    for i, record in enumerate(records):
        got, c = samples(record, form)
        rows += [(i, *r) for r in got]
        labels.add("exposed" in c)
        for k, v in c.items():
            cut[k] = cut.get(k, 0) + v
    if len(labels) > 1:
        raise ValueError(
            "some records carry device windows and some do not; one law takes one label"
        )
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
                    "centre": solved["centre"],
                    "mad": solved["mad"],
                    "zero": solved["zero"],
                },
            }
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
    if "host" in law:
        print(f"host: {json.dumps(law['host']['laws'])} {json.dumps(law['host_fit'])}")
    if len(args.records) > 1:
        for form in FORMS:
            print(f"\nheld out, form {form}\n{_table(held_out(args.records, form))}")
        if "host" in law:
            print(f"\nheld out, host laws\n{_table(host_held_out(args.records))}")


if __name__ == "__main__":
    main()
