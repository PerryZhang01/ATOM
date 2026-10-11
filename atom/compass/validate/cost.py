# SPDX-License-Identifier: MIT
"""Cost accuracy: a law priced on a real run's own step sequence.

    python -m atom.compass.validate.cost RECORD LAW

RECORD is a real run's step record directory (`ATOM_COMPASS_PARITY_RECORD`,
`OUT_DIR/steps` of a `REAL=1 cctraces_sim.sh` cell); LAW is a law file as
`coarse_fit` writes it. Pricing the real run's own steps separates cost error
from schedule error: both sides run the same batches.

Each step is labelled as the fit labels it, by its period (`coarse_fit.samples`,
which also says which steps carry no label), and priced by `CoarseBackend`. The
error of a step is `priced / measured - 1`. Per kind the report gives the error
on totals, on totals holding out the step that contributes most to their error,
and the median per step; per decode rung or prefill token band, the p10, p50
and p90 of the per-step error and the fraction of steps within 10%.

A step outside its vector's hull is priced, but its error describes the
extrapolation, not the law: those steps are reported apart, with their measured
and priced seconds, and enter none of the in-hull figures. A step that ran more
than `coarse_fit.WARM_FACTOR` times its price is warm, a one-off first touch the
law does not model: it is named with its seconds and enters no figure. So are
the steps that lead a vector's priced steps, each slower than its price by more
than `coarse_fit.MAD_K` scaled MADs of that vector's errors, as the fit names
them. A step the law refuses is counted with its measured seconds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from atom.compass import parity
from atom.compass.backends import (
    BatchView,
    CoarseBackend,
    CoarseLaw,
    CostRefused,
    RequestShape,
    Species,
)
from atom.compass.backends.coarse_fit import (
    MAD_FLOOR,
    MAD_K,
    WARM_FACTOR,
    _band,
    samples,
)

WITHIN = 0.10


def _leading_slow(priced: list[tuple]) -> set[int]:
    """Steps leading their vector's (vector, band, step, seconds, cost) rows, each slow."""
    out = set()
    for key in sorted({p[0] for p in priced}):
        mine = [p for p in priced if p[0] == key]
        error = np.array([p[4].seconds / p[3] - 1.0 for p in mine])
        centre = np.median(error)
        mad = 1.4826 * np.median(np.abs(error - centre))
        slow = (error < 0) & (np.abs(error - centre) > max(MAD_K * mad, MAD_FLOOR))
        for p, s in zip(mine, slow):
            if not s:
                break
            out.add(p[2])
    return out


def _summary(steps: list[tuple]) -> dict:
    """(band, step, measured, priced) rows of one kind as the report's figures."""
    measured = np.array([s[2] for s in steps])
    priced = np.array([s[3] for s in steps])
    error = priced / measured - 1.0
    largest = int(np.argmax(np.abs(priced - measured)))
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
        "measured_s": float(measured.sum()),
        "priced_s": float(priced.sum()),
        "on_totals": float(priced.sum() / measured.sum() - 1.0),
        "held_out_step": steps[largest][1],
        "held_out": (
            float(priced[rest].sum() / measured[rest].sum() - 1.0)
            if rest.any()
            else None
        ),
        "median": float(np.median(error)),
        "bands": bands,
    }


def cost_accuracy(record, law) -> dict:
    """The report for `law`, a mapping `CoarseLaw` reads, on the record directory `record`."""
    ranks = parity.read(record)
    if len(ranks) != 1:
        raise ValueError(
            f"{record} holds {len(ranks)} DP ranks' records; this prices one rank's"
        )
    ((rank, steps),) = ranks.items()
    labelled, cut = samples(Path(record) / f"dp{rank}.jsonl")
    by_step = {s["step"]: s for s in steps}
    backend = CoarseBackend(CoarseLaw(law))
    groups = {"fitted": {}, "extrapolated": {}}
    refused: dict[str, dict] = {}
    warm, priced = [], []
    for key, counts, seconds, step in labelled:
        line = by_step[step]
        view = BatchView(
            tuple(RequestShape(q, c, bool(d)) for q, c, d in line["rows"]),
            capture_rung=line["rung"],
        )
        band = _band(key, counts)
        try:
            cost = backend.estimate(view)
        except CostRefused:
            r = refused.setdefault(band, {"steps": 0, "measured_s": 0.0})
            r["steps"] += 1
            r["measured_s"] += seconds
            continue
        priced.append((key, band, step, seconds, cost))
    leading = _leading_slow(priced)
    for key, band, step, seconds, cost in priced:
        if step in leading or seconds > WARM_FACTOR * cost.seconds:
            warm.append(
                {
                    "step": step,
                    "band": band,
                    "measured_s": seconds,
                    "priced_s": cost.seconds,
                }
            )
            continue
        species = {t.provenance.species for t in cost.terms}
        group = "extrapolated" if Species.EXTRAPOLATED in species else "fitted"
        kind = "prefill" if key == "prefill" else "decode"
        groups[group].setdefault(kind, []).append((band, step, seconds, cost.seconds))
    report = {"record": str(record), "law": backend.law.provenance, "cut": cut}
    for group, kinds in groups.items():
        report[group] = {
            kind: _summary(rows) for kind, rows in sorted(kinds.items(), reverse=True)
        }
    report["warm"] = warm
    report["refused"] = refused
    return report


def _text(report: dict) -> str:
    out = [f"record {report['record']}", f"law {report['law']}"]
    out.append(f"not labelled: {json.dumps(report['cut'])}")
    for group, title in (
        ("fitted", "inside the hull"),
        ("extrapolated", "outside the hull, apart"),
    ):
        out.append(f"\n{title}")
        for kind, s in report[group].items():
            held = "-" if s["held_out"] is None else f"{s['held_out']:+.2%}"
            out += [
                (
                    f"{kind}  {s['steps']} steps  measured {s['measured_s']:.2f} s"
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
            f"\nwarm step {w['step']} ({w['band']}): measured {w['measured_s']:.2f} s,"
            f" priced {w['priced_s']:.2f} s"
        )
    for band, r in report["refused"].items():
        out.append(
            f"\nrefused {band}: {r['steps']} steps, measured {r['measured_s']:.2f} s"
        )
    return "\n".join(out)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m atom.compass.validate.cost",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("record", help="a real run's step record directory")
    parser.add_argument("law", help="a law file, as coarse_fit writes it")
    args = parser.parse_args(argv)
    print(_text(cost_accuracy(args.record, json.loads(Path(args.law).read_text()))))


if __name__ == "__main__":
    main()
