# SPDX-License-Identifier: MIT
"""Cost accuracy: a law, or a step grid, priced on a real run's own step sequence.

    python -m atom.compass.validate.cost RECORD LAW

RECORD is a real run's step record directory (`ATOM_COMPASS_PARITY_RECORD`,
`OUT_DIR/steps` of a `REAL=1 cctraces_sim.sh` cell); LAW is a law file as
`step_linear_fit` writes it. Pricing the real run's own steps separates cost error
from schedule error: both sides run the same batches. The command prices every
step with a law; `python -m atom.compass.validate` prices only those of the client
run it grades, its warmup left out, with the backend the simulated run priced with.

Each step is labelled as the fit labels it, by its device time
(`step_labels.label`, which also says which steps carry no label and refuses a
record without device windows), and priced by `StepLinearBackend` or `StepGridBackend`. The
error of a step is `priced / real - 1`, its target price over its target label on
an MTP record. A step that also carries a draft label is graded again under kind
`draft`, its draft price over that label, validated only if it has draft terms and
each is. Per kind the report gives the error
on totals, on totals holding out the step that contributes most to their error,
and the median per step; per decode rung or prefill token band, the p10, p50
and p90 of the per-step error and the fraction of steps within 10%.

A step whose terms are not validated (outside the hull of the law's held-out
steps, every step of a law that records none; for the grid, outside its
residual's held-out hull) is priced, but its error was
not measured where the law's was: those steps are reported apart, with their
real and priced seconds, and enter none of the validated figures. A step that ran more
than `step_linear_fit.WARM_FACTOR` times its price is warm, a one-off first touch the
law does not model: it is named with its seconds and enters no figure. So are
the steps that lead a vector's priced steps, each slower than its price by more
than `step_linear_fit.MAD_K` scaled MADs of the errors the law's vector was fitted
on, as the fit names them; of that vector's errors here if the law records no
`fit.mad`, as the grid never does. A step the backend declines is counted with
its real seconds and its reasons, the most frequent named: under `fallback` for
the grid, since its run prices the step by the step roofline below it, and under
`refused` for a law, since a run grading takes has nothing below the law. A step
it extrapolated, past its support domain, is counted too.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from atom.compass.backends import (
    CostRefused,
    StepGridBackend,
    StepLinearBackend,
    StepLinearLaw,
    step_labels,
    step_linear_fit,
)
from atom.compass.backends.step_linear_fit import WARM_FACTOR, _band, _groups

WITHIN = 0.10
REASONS = 3  # a declined band names this many of its most frequent reasons


def _leading_slow(priced: list[tuple], law: dict | None) -> set[int]:
    """Steps leading their vector's (vector, band, step, seconds, price) rows, each
    slow by the scatter `law` fitted the vector on, or by the scatter here where
    `law` records none or is None, as for a step grid."""
    out = set()
    for key in sorted({p[0] for p in priced}):
        mine = [p for p in priced if p[0] == key]
        error = np.array([p[4] / p[3] - 1.0 for p in mine])
        centre = np.median(error)
        mad = 1.4826 * np.median(np.abs(error - centre))
        vector = {}
        if law is not None:
            vector = law["prefill"] if key == "prefill" else law["decode"][key]
        if "mad" in vector.get("fit", {}):
            centre, mad = vector["fit"]["centre"], vector["fit"]["mad"]
        slow = (error < 0) & (
            np.abs(error - centre)
            > max(step_linear_fit.MAD_K * mad, step_linear_fit.MAD_FLOOR)
        )
        for p, s in zip(mine, slow):
            if not s:
                break
            out.add(p[2])
    return out


def _summary(steps: list[tuple]) -> dict:
    """(band, step, real, priced) rows of one kind as the report's figures."""
    real = np.array([s[2] for s in steps])
    priced = np.array([s[3] for s in steps])
    error = priced / real - 1.0
    largest = int(np.argmax(np.abs(priced - real)))
    rest = np.arange(len(steps)) != largest
    bands = {}
    for band in sorted({s[0] for s in steps}):
        e = error[[s[0] == band for s in steps]]
        p10, p50, p90 = np.percentile(e, (10, 50, 90))
        bands[band] = {
            "steps": len(e),
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "within_10pct": float(np.mean(np.abs(e) <= WITHIN)),
        }
    return {
        "steps": len(steps),
        "real_s": float(real.sum()),
        "priced_s": float(priced.sum()),
        "on_totals": float(priced.sum() / real.sum() - 1.0),
        "held_out_step": steps[largest][1],
        "held_out": (
            float(priced[rest].sum() / real[rest].sum() - 1.0) if rest.any() else None
        ),
        "median": float(np.median(error)),
        "bands": bands,
    }


def cost_accuracy(record, backend, steps=None) -> dict:
    """The report for `backend`, a `CostBackend` or a law mapping
    `StepLinearLaw` reads, on the record directory `record`: on its steps whose
    numbers are in `steps`, every step when None. Its ``backend`` names the
    law's provenance, or the backend's `describe()` line.

    The ranks' records (`dp0.jsonl` up) are joined as `step_linear_fit` joins
    them: a step is its ranks' lines of one step index, measured as the
    smallest rank label and priced and banded as its costliest rank's batch, a
    term validated only if every rank's is; a step the law refuses is banded as
    its rank of most tokens. Its draft is measured and priced the same way. A step some rank lacks or cut, or whose
    ranks fall to different vectors, is counted under `cut.ranks`. Ranks whose
    records differ in length by more than the one step a stopped run leaves on
    some ranks only are refused.
    """
    ranks = sorted(Path(record).glob("dp*[0-9].jsonl"))
    lines = {r.name: len(r.read_text().splitlines()) for r in ranks}
    if not ranks or max(lines.values()) - min(lines.values()) > 1:
        raise ValueError(
            f"{record} holds rank records of {lines} lines; one run's ranks "
            "differ by at most the step a stopped run leaves on some"
        )
    run = [
        [s for s in step_labels.label(r)[0] if steps is None or s.step in steps]
        for r in ranks
    ]
    joined, cut, _ = _groups(run, "base")
    views = [{s.step: s.view for s in rank} for rank in run]
    drafts = [{s.step: s.draft_s for s in rank} for rank in run]
    law = backend if isinstance(backend, Mapping) else None
    if law is not None:
        backend = StepLinearBackend(StepLinearLaw(law))
    groups = {"validated": {}, "unvalidated": {}}
    declined: dict[str, dict] = {}
    warm, priced, extrapolated = [], [], 0
    for group in joined:
        key, _, seconds, step, _ = group[0]
        try:
            costs = [backend.price(views[i][step]) for i in range(len(group))]
        except CostRefused as refused:
            band = _band(key, max((r[1] for r in group), key=lambda c: c[1]))
            r = declined.setdefault(
                band, {"steps": 0, "real_s": 0.0, "reasons": Counter()}
            )
            r["steps"] += 1
            r["real_s"] += seconds
            why = "; ".join(f"{d.source}: {d.reason}" for d in refused.declined)
            r["reasons"][why] += 1
            continue
        terms = [t for c in costs for t in c.terms if not t.draft]
        validated = all(t.provenance.validated for t in terms)
        extrapolated += any(t.provenance.extrapolated for t in terms)
        top = max(range(len(costs)), key=lambda i: costs[i].target_seconds)
        band, price = _band(key, group[top][1]), costs[top].target_seconds
        labels = [d[step] for d in drafts]
        draft = None
        if None not in labels:
            mine = [t.provenance.validated for t in costs[top].terms if t.draft]
            draft = (min(labels), costs[top].draft_seconds, bool(mine) and all(mine))
        priced.append((key, band, step, seconds, price, validated, draft))
    leading = _leading_slow(priced, law)
    for key, band, step, seconds, price, validated, draft in priced:
        if step in leading or seconds > WARM_FACTOR * price:
            warm.append(
                {
                    "step": step,
                    "band": band,
                    "real_s": seconds,
                    "priced_s": price,
                }
            )
            continue
        group = "validated" if validated else "unvalidated"
        kind = "prefill" if key == "prefill" else "decode"
        groups[group].setdefault(kind, []).append((band, step, seconds, price))
        if draft is not None:
            group = "validated" if draft[2] else "unvalidated"
            groups[group].setdefault("draft", []).append((band, step, *draft[:2]))
    name = backend.describe() if law is None else backend.law.fitted_on
    report = {"record": str(record), "backend": name, "cut": cut}
    for group, kinds in groups.items():
        report[group] = {
            kind: _summary(rows) for kind, rows in sorted(kinds.items(), reverse=True)
        }
    report["warm"] = warm
    grid = isinstance(backend, StepGridBackend)
    report["fallback"], report["refused"] = (declined, {}) if grid else ({}, declined)
    report["extrapolated"] = extrapolated
    return report


def declined_text(r: dict) -> str:
    """A declined band's steps, real seconds and most frequent reasons."""
    reasons = r["reasons"].most_common()
    named = json.dumps(dict(reasons[:REASONS]))
    more = f" and {len(reasons) - REASONS} more" if len(reasons) > REASONS else ""
    return f"{r['steps']} steps, real {r['real_s']:.2f} s; reasons {named}{more}"


def _text(report: dict) -> str:
    out = [f"record {report['record']}", f"law {report['backend']}"]
    out.append(f"not labelled: {json.dumps(report['cut'])}")
    for group, title in (
        ("validated", "validated: inside the held-out hull"),
        ("unvalidated", "unvalidated, apart"),
    ):
        out.append(f"\n{title}")
        for kind, s in report[group].items():
            held = "-" if s["held_out"] is None else f"{s['held_out']:+.2%}"
            out += [
                (
                    f"{kind}  {s['steps']} steps  real {s['real_s']:.2f} s"
                    f"  priced {s['priced_s']:.2f} s"
                ),
                f"  on totals              {s['on_totals']:+.2%}",
                f"  holding out step {s['held_out_step']:<6}{held}",
                f"  median per step        {s['median']:+.2%}",
                "  band | steps | p10 | p50 | p90 | within 10%",
            ]
            for band, b in s["bands"].items():
                out.append(
                    f"  {band} | {b['steps']} | {b['p10']:+.1%} | {b['p50']:+.1%}"
                    f" | {b['p90']:+.1%} | {b['within_10pct']:.1%}"
                )
    for w in report["warm"]:
        out.append(
            f"\nwarm step {w['step']} ({w['band']}): real {w['real_s']:.2f} s,"
            f" priced {w['priced_s']:.2f} s"
        )
    for band, r in report["refused"].items():
        out.append(f"\nrefused {band}: {declined_text(r)}")
    return "\n".join(out)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m atom.compass.validate.cost",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("record", help="a real run's step record directory")
    parser.add_argument("law", help="a law file, as step_linear_fit writes it")
    args = parser.parse_args(argv)
    print(_text(cost_accuracy(args.record, json.loads(Path(args.law).read_text()))))


if __name__ == "__main__":
    main()
