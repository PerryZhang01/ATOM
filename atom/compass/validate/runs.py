# SPDX-License-Identifier: MIT
"""Schedule agreement and end-to-end results: a simulated cell against N real runs of it.

    python -m atom.compass.validate.runs SIMULATED REAL...

Each argument is a run directory: its step record in `steps/` and aiperf's
export in `artifacts/profile_export_aiperf.json`. SIMULATED is the cell at the
law, and REAL two or more real runs of it, spaced in time.

A directory `cctraces_sim.sh` wrote is graded whole, and refused when its
run.json skips host segments. In one `record` or `run` wrote, whose run.json
states a `format_version`, the graded client run is the one whose export is
`artifacts/client_run.json`, split from the record by `parity.served`: only
its steps are graded, and every run must have served the same traces. It is
refused, naming every reason, when steps of requests it did not send precede
its own: it is then not its server's first client run, and meets the prefix
cache an earlier one left. A simulated run, whose run.json holds `run_file`,
is also refused when it skips host segments, or its run file has no host laws
and so prices no host time; when the step roofline was on; when the run
refused anything; when it ran the step grid without a passed coverage check;
and when its backend's fit inputs were profiled or served a trace it served.

The tolerance of a metric is the real runs' own spread: its distance between
every ordered pair of real runs, at most. A metric's result is its distance
from the simulated cell to each real run. The families:

- counts, from the step record and the export: steps and tokens per kind,
  preemptions, prefill streaks, requests completed. Distance `|a - b|`.
  `exact` where every real run has the simulated cell's value. `prefill
  tokens` counts only the requests both runs issued, since which turns a
  cc-traces run issues before its cutoff moves with its end-to-end time; the
  values it reports are the cell totals.
- one run, `prefill tokens`: the prefill tokens of the requests only the first
  run issued less those only the second did, `not gated`. A request is its
  prompt length and its place in arrival order among the run's requests of
  that length, which a prefix-cache hit does not move; the requests a record
  never names, their prefill unfinished, are one request.
- distributions, per step from the step record: batch width, scheduled
  tokens, context tokens, and the cached fraction of a prefill step's context.
  Distance: the two-sample KS statistic.
- end to end, from the export: TTFT, TPOT (aiperf's inter-token latency) mean
  and p50, request and output-token throughput. Distance `|a / b - 1|`.
  `ungradeable` where the real spread exceeds the target (15% TTFT, 10% the
  others); else `within target` where every result is inside it.

Every metric is `inside the real spread` where each result is at most the
spread, else `outside`.

A step is prefill when any of its rows is. A preemption is a request's row
that is prefill after one of its decode rows, or that reads no more context
than its previous row did.
"""

from __future__ import annotations

import argparse
import collections
import json
from itertools import permutations
from pathlib import Path

from scipy.stats import ks_2samp

from atom.compass.parity.served import served
from atom.compass.run import STEP_GRID

EXPORT = "artifacts/profile_export_aiperf.json"
#: The export of the client run a `record` or `run` directory grades.
CLIENT_RUN = "artifacts/client_run.json"
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
    # Each rank's record, and not the cost lines a simulated run writes beside it.
    ranks = sorted((Path(cell) / "steps").glob("dp*[0-9].jsonl"))
    if len(ranks) != 1:
        raise ValueError(
            f"{cell} holds {len(ranks)} DP ranks' records; this grades one"
        )
    steps = (json.loads(line) for line in ranks[0].read_text().splitlines())
    return [s for s in steps if not s["is_dummy_run"]]


def _requests(steps) -> collections.Counter:
    """Prefill tokens per request, keyed by (prompt length, arrival place)."""
    tokens, length = collections.Counter(), {}
    for step in steps:
        named = {request for request, _key, *_response_id in step["named"]}
        for (request, _n, context), (q, _c, decode) in zip(step["batch"], step["rows"]):
            tokens[request] += 0 if decode else q
            if request in named:  # its final chunk: the context is the prompt
                length[request] = context
    out, seen = collections.Counter(), collections.Counter()
    for request in sorted(tokens):  # request ids number arrivals
        n = length.get(request)
        out[None if n is None else (n, seen[n])] += tokens[request]
        seen[n] += 1
    return out


def _both(a, b) -> int:
    return abs(sum(n - b[k] for k, n in a.items() if k in b))


def _one_run(a, b) -> int:
    only_a = sum(n for k, n in a.items() if k not in b)
    return abs(only_a - sum(n for k, n in b.items() if k not in a))


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


def _client_run(cell, run) -> tuple[list[dict], list[str]]:
    """The steps of the client run `cell` grades and the traces they served;
    refuses, naming every reason, a run grading does not take."""
    steps = _steps(cell)
    (split,) = served(Path(cell) / "steps", [Path(cell) / CLIENT_RUN]).values()
    mine = [s["client_run"] == 0 for s in split if not s["is_dummy_run"]]
    traces = sorted({t for s in split if s["client_run"] == 0 for t in s["traces"]})
    why = [] if mine[0] else ["steps of requests it did not send precede its own"]
    if "run_file" in run:
        why += _not_graded(run, traces)
    if why:
        raise ValueError(f"{cell} is not graded: " + "; ".join(why))
    return [s for s, m in zip(steps, mine) if m], traces


def _not_graded(run, traces) -> list[str]:
    """Why a simulated run, `run` its run.json, whose client run served
    `traces`, is not graded."""
    why, refused, coverage = [], run["refusals"], run["coverage"]
    if run["skip"]:
        why.append(f"it skips host segments {run['skip']}, a what-if run")
    if "host" not in run["run_file"]:
        why.append("its run file has no host laws, so it prices no host time")
    if run["step_roofline"]:
        why.append("the step roofline was on")
    if refused["count"]:
        why.append(f"it refused {refused['count']}: {refused['reasons']}")
    if STEP_GRID in run["run_file"] and not (coverage and not coverage["uncovered"]):
        why.append("it ran the step grid without a passed coverage check")
    inputs = (run["provenance"]["provenance"] or {}).get("inputs") or []
    if fitted := sorted({t for i in inputs for t in i["traces"]} & set(traces)):
        why.append(f"its backend was fitted on traces it served: {fitted}")
    if profiled := [i["run"] for i in inputs if i.get("profiled")]:
        why.append(f"its backend was fitted on profiled inputs: {profiled}")
    return why


def _read(cell) -> dict:
    path = Path(cell) / "run.json"
    run = json.loads(path.read_text()) if path.exists() else {}
    if "format_version" in run:
        steps, traces = _client_run(cell, run)
    elif run.get("host", {}).get("skip"):
        raise ValueError(f"{cell} skips host segments; a what-if run is not graded")
    else:
        steps, traces = _steps(cell), None
    export = json.loads((Path(cell) / EXPORT).read_text())
    return {
        "counts": _counts(steps, export),
        "distributions": _distributions(steps),
        "end to end": _end_to_end(export),
        "requests": _requests(steps),
        "traces": traces,
    }


DISTANCE = {
    "counts": lambda a, b: abs(a - b),
    "distributions": lambda a, b: ks_2samp(a, b).statistic,
    "end to end": lambda a, b: abs(a / b - 1.0),
}


def _grade(family, name, real, simulated, d=None) -> dict:
    d = d or DISTANCE[family]
    spread = max(d(a, b) for a, b in permutations(real, 2))
    result = [d(simulated, r) for r in real]
    out = {"spread": spread, "result": result}
    if family != "distributions":
        out.update(real=real, simulated=simulated)
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
    return out


def grade(real_cells, simulated_cell) -> dict:
    """Every metric's grade: {family: {metric: grade}}, as the module says."""
    if len(real_cells) < 2:
        raise ValueError("a real spread needs two or more real runs")
    real = [_read(c) for c in real_cells]
    simulated = _read(simulated_cell)
    sent = {str(c): r["traces"] for c, r in zip(real_cells, real)}
    sent[str(simulated_cell)] = simulated["traces"]
    if len({json.dumps(t) for t in sent.values()}) > 1:
        every = set.intersection(*(set(t or ()) for t in sent.values()))
        odd = {c: sorted(set(t or ()) - every) for c, t in sent.items()}
        raise ValueError(
            f"the runs sent different traces; beyond those all sent: {odd}"
        )
    report = {
        family: {
            name: _grade(
                family, name, [r[family][name] for r in real], simulated[family][name]
            )
            for name in simulated[family]
        }
        for family in DISTANCE
    }
    totals = {k: report["counts"]["prefill tokens"][k] for k in ("real", "simulated")}
    requests = [r["requests"] for r in real], simulated["requests"]
    report["counts"]["prefill tokens"] = (
        _grade("counts", "prefill tokens", *requests, _both) | totals
    )
    one_run = _grade("counts", "prefill tokens", *requests, _one_run) | totals
    report["one run"] = {"prefill tokens": one_run | {"verdict": "not gated"}}
    return report


def _text(report: dict) -> str:
    out = ["family | metric | spread | result | verdict"]
    for family, metrics in report.items():
        for name, g in metrics.items():
            result = " ".join(f"{r:.4g}" for r in g["result"])
            out.append(
                f"{family} | {name} | {g['spread']:.4g} | {result} | {g['verdict']}"
            )
    return "\n".join(out)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m atom.compass.validate.runs",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("simulated", help="the simulated cell at the law")
    parser.add_argument("real", nargs="+", help="two or more real runs of the cell")
    args = parser.parse_args(argv)
    print(_text(grade(args.real, args.simulated)))


if __name__ == "__main__":
    main()
