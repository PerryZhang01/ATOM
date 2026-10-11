# SPDX-License-Identifier: MIT
"""Schedule agreement and end-to-end results: a simulated cell against N real runs of it.

    python -m atom.compass.validate.runs SIMULATED PERTURBED REAL...

Each argument is a cell directory of `cctraces_sim.sh`: its step record in
`steps/` and aiperf's export in `artifacts/profile_export_aiperf.json`.
SIMULATED is the cell at the law, PERTURBED the same cell at 0.97 of the law's
prefill price, and REAL two or more real runs of it, spaced in time.

The tolerance of a metric is the real runs' own spread: its distance between
every ordered pair of real runs, at most. A metric's result is its distance
from the simulated cell to each real run. The families:

- counts, from the step record and the export: steps and tokens per kind,
  preemptions, prefill streaks, requests completed. Distance `|a - b|`.
  `exact` where every real run has the simulated cell's value.
- distributions, per step from the step record: batch width, scheduled
  tokens, context tokens, and the cached fraction of a prefill step's context.
  Distance: the two-sample KS statistic.
- end to end, from the export: TTFT, TPOT (aiperf's inter-token latency) mean
  and p50, request and output-token throughput. Distance `|a / b - 1|`.
  `ungradeable` where the real spread exceeds the target (15% TTFT, 10% the
  others); else `within target` where every result is inside it.

Every metric is `inside the real spread` where each result is at most the
spread, else `outside`. One that moves no further from the simulated cell
under the perturbed price than the real spread is `inadmissible`: it cannot
grade a cost model, and neither can an `ungradeable` one.

A step is prefill when any of its rows is. A preemption is a request's row
that is prefill after one of its decode rows, or that reads no more context
than its previous row did.
"""

from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path

from scipy.stats import ks_2samp

from atom.compass import parity

EXPORT = "artifacts/profile_export_aiperf.json"
TARGET = {"ttft": 0.15}
DEFAULT_TARGET = 0.10
END_TO_END = {
    "ttft mean": ("time_to_first_token", "avg"),
    "ttft p50": ("time_to_first_token", "p50"),
    "tpot mean": ("inter_token_latency", "avg"),
    "tpot p50": ("inter_token_latency", "p50"),
    "request throughput": ("request_throughput", "avg"),
    "output token throughput": ("output_token_throughput", "avg"),
}


def _steps(cell) -> list[dict]:
    ranks = parity.read(Path(cell) / "steps")
    if len(ranks) != 1:
        raise ValueError(
            f"{cell} holds {len(ranks)} DP ranks' records; this grades one"
        )
    return [s for s in next(iter(ranks.values())) if not s["is_dummy_run"]]


def _prefill(step) -> bool:
    return not all(decode for _q, _c, decode in step["rows"])


def _counts(steps, export) -> dict:
    out = dict.fromkeys(
        (
            "prefill steps",
            "decode steps",
            "prefill tokens",
            "decode tokens",
            "preemptions",
            "prefill streaks",
        ),
        0,
    )
    last, was_prefill = {}, False
    for step in steps:
        prefill = _prefill(step)
        out["prefill steps" if prefill else "decode steps"] += 1
        out["prefill streaks"] += int(prefill and not was_prefill)
        was_prefill = prefill
        for (request, _n, _c), (q, context, decode) in zip(step["batch"], step["rows"]):
            out[f"{'decode' if decode else 'prefill'} tokens"] += q
            before = last.get(request)
            if before and not decode and (before[1] or context <= before[0]):
                out["preemptions"] += 1
            last[request] = (context, decode)
    out["requests completed"] = export["request_count"]["avg"]
    return out


def _distributions(steps) -> dict:
    cached = [
        sum(c - q for q, c, _ in s["rows"]) / sum(c for _, c, _ in s["rows"])
        for s in steps
        if _prefill(s)
    ]
    return {
        "batch width": [len(s["rows"]) for s in steps],
        "scheduled tokens": [sum(q for q, _, _ in s["rows"]) for s in steps],
        "context tokens": [sum(c for _, c, _ in s["rows"]) for s in steps],
        "cached fraction": cached,
    }


def _end_to_end(export) -> dict:
    return {name: export[k][stat] for name, (k, stat) in END_TO_END.items()}


def _read(cell) -> dict:
    run = Path(cell) / "run.json"
    if run.exists() and json.loads(run.read_text()).get("host", {}).get("skip"):
        raise ValueError(f"{cell} skips host segments; a what-if run is not graded")
    steps = _steps(cell)
    export = json.loads((Path(cell) / EXPORT).read_text())
    return {
        "counts": _counts(steps, export),
        "distributions": _distributions(steps),
        "end to end": _end_to_end(export),
    }


DISTANCE = {
    "counts": lambda a, b: abs(a - b),
    "distributions": lambda a, b: ks_2samp(a, b).statistic,
    "end to end": lambda a, b: abs(a / b - 1.0),
}


def _grade(family, name, real, simulated, perturbed) -> dict:
    d = DISTANCE[family]
    spread = max(d(a, b) for a, b in permutations(real, 2))
    result = [d(simulated, r) for r in real]
    moved = d(perturbed, simulated)
    out = {"spread": spread, "result": result, "moved": moved}
    if family != "distributions":
        out.update(real=real, simulated=simulated, perturbed=perturbed)
    target = TARGET.get(name.split()[0], DEFAULT_TARGET)
    if family == "counts" and max(result) == 0:
        verdict = "exact"
    elif family == "end to end" and spread > target:
        verdict = "ungradeable"
    elif max(result) <= spread:
        verdict = "inside the real spread"
    elif family == "end to end" and max(result) <= target:
        verdict = "within target"
    else:
        verdict = "outside"
    out["verdict"] = verdict
    out["admissible"] = moved > spread and verdict != "ungradeable"
    return out


def grade(real_cells, simulated_cell, perturbed_cell) -> dict:
    """Every metric's grade: {family: {metric: grade}}, as the module says."""
    if len(real_cells) < 2:
        raise ValueError("a real spread needs two or more real runs")
    real = [_read(c) for c in real_cells]
    simulated, perturbed = _read(simulated_cell), _read(perturbed_cell)
    return {
        family: {
            name: _grade(
                family,
                name,
                [r[family][name] for r in real],
                simulated[family][name],
                perturbed[family][name],
            )
            for name in simulated[family]
        }
        for family in DISTANCE
    }


def _text(report: dict) -> str:
    out = ["family | metric | spread | result | moved | verdict | admissible"]
    for family, metrics in report.items():
        for name, g in metrics.items():
            result = " ".join(f"{r:.4g}" for r in g["result"])
            out.append(
                f"{family} | {name} | {g['spread']:.4g} | {result} | {g['moved']:.4g}"
                f" | {g['verdict']} | {'yes' if g['admissible'] else 'no'}"
            )
    return "\n".join(out)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m atom.compass.validate.runs",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("simulated", help="the simulated cell at the law")
    parser.add_argument(
        "perturbed", help="the simulated cell at 0.97 of its prefill price"
    )
    parser.add_argument("real", nargs="+", help="two or more real runs of the cell")
    args = parser.parse_args(argv)
    print(_text(grade(args.real, args.simulated, args.perturbed)))


if __name__ == "__main__":
    main()
