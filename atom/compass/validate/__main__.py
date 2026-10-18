# SPDX-License-Identifier: MIT
"""Grade cells from every source, in one table: `python -m atom.compass.validate CELL...`

A CELL holds runs of one workload: `step_linear/` (at its run file's step linear law, or
of several the one fitted under the engine config each real run states)
and two or more `real*/`. `cctraces_sim.sh` writes each run.json as the run file with
the `workload` every run must share; `record` and `run` write their own, the run file
as `run_file`, and `validate.runs` decides which client run of theirs is graded, and
whether one is. Rows: `validate.runs`' grades, TTFT p50 per band of uncached prompt
tokens graded alike; per run `record` or `run` wrote, its warmup: how many requests
its export names as such and how many steps hold only those; `validate.cost` on the
steps graded (with the steps it leaves out) and `memory.check` per real run; and the
simulated run's refusals by source, its steps extrapolated and priced by a
fallback, shares and speed. Any refusal makes a
`cctraces_sim.sh` cell a coverage report; the shares grade nothing, and neither do the
counts but preemptions and prefill tokens.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import median

from atom.compass.backends.cost import GROUPS
from atom.compass.memory.check import MemoryRefusal, parts
from atom.compass.run import (
    LP_FILE,
    STEP_LINEAR,
    _fitted_here,
    entries,
    refuse_renamed_keys,
)
from atom.compass.run.step_grid import keyed_of
from atom.compass.validate import cost, runs

BANDS = {0: "0-512", 512: "512-2048", 2048: "2048-8192", 8192: "8192+"}
HULL = {"validated": ("validated", "-"), "unvalidated": ("unvalidated", "not gated")}
GATED_COUNTS = {"preemptions", "prefill tokens"}


def _ttft(run: Path) -> dict[str, float]:
    by = {}
    for line in (run / "artifacts/profile_export.jsonl").read_text().splitlines():
        r = json.loads(line)
        m, phase = r["metrics"], r["metadata"]["benchmark_phase"]
        if phase == "profiling" and "time_to_first_token" in m:
            n = m["usage_prompt_tokens"]["value"]
            n -= m["usage_prompt_cache_read_tokens"]["value"]
            band = BANDS[max(e for e in BANDS if e <= n)]
            by.setdefault(band, []).append(m["time_to_first_token"]["value"])
    return {band: median(v) for band, v in by.items()}


def _graded(family: str, name: str, g: dict) -> tuple:
    result = " ".join(f"{r:.4g}" for r in g["result"])
    value, verdict = f"spread {g['spread']:.4g}; result {result}", g["verdict"]
    if family == "counts" and name not in GATED_COUNTS:
        verdict += ", not gated"
    return family, name, value, verdict


def _runs(real: list[Path], law: Path):
    for family, metrics in runs.grade(real, law).items():
        for name, g in metrics.items():
            yield _graded(family, name, g)
    every = (*real, law)
    ttft = [_ttft(r) for r in every]
    for band in BANDS.values():
        v, name = [t.get(band) for t in ttft], f"ttft p50 {band} uncached"
        if None in v:
            missing = [r.name for r, x in zip(every, v) if x is None]
            yield "end to end", name, f"no request in {missing}", "-"
        else:
            g = runs._grade("end to end", "ttft", v[:-1], v[-1])
            yield _graded("end to end", name, g)


def _warmup(run: Path, warmup: dict | None):
    if warmup is not None:  # None: a run `cctraces_sim.sh` wrote, graded whole
        value = f"requests {warmup['requests']}, steps {warmup['steps']}"
        yield f"warmup {run.name}", "warmup", value, "not graded"


def _cost(run: Path, law: dict, steps: list[dict]):
    numbers = {s["step"] for s in steps}
    report, where = cost.cost_accuracy(run / "steps", law, numbers), f"cost {run.name}"
    for group, (hull, gated) in HULL.items():
        for kind, s in report[group].items():
            held = "-" if s["held_out"] is None else f"{s['held_out']:+.2%}"
            value = (
                f"{s['steps']} steps, real {s['real_s']:.2f} s, priced {s['priced_s']:.2f} s;"
                f" on totals {s['on_totals']:+.2%}, holding out step {s['held_out_step']} {held},"
                f" median {s['median']:+.2%}"
            )
            yield where, f"{kind} {hull}", value, gated
            for band, b in s["bands"].items():
                value = f"{b['steps']} steps; p10 {b['p10']:+.1%} p50 {b['p50']:+.1%} p90 {b['p90']:+.1%}"
                yield where, f"{band} {hull}", f"{value}; within 10% {b['within_10pct']:.1%}", gated
    yield where, "left out", f"not labelled {json.dumps(report['cut'])}; warm steps, real s { {w['step']: round(w['real_s'], 2) for w in report['warm']} }", "-"
    for band, r in report["refused"].items():
        yield where, f"refused {band}", f"{r['steps']} steps, real {r['real_s']:.2f} s", "-"


def _memory(law: Path, real: Path):
    where = f"memory {real.name}"
    try:
        comparison, *_, graph = parts(
            law / LP_FILE.format("engine"), real / "server.log"
        )
    except MemoryRefusal as e:
        yield where, "memory", e.what, "refused"
        return
    for t in comparison.compared:
        yield where, t.name, f"{t.relative:+.2%} ({t.delta_bytes:+,} B)", str(t.verdict)
    for r in comparison.refused:
        yield where, r.name, r.what, "refused"
    if not isinstance(graph, str):
        graph = f"reserves() {graph.reserves_relative:+.2%}, predicts() {graph.predicts_relative:+.2%}"
    yield where, "graph pool", graph, "-"


def _summary(run: Path, summary: dict):
    schedule, c, where = summary["schedule"], summary["cost"], f"summary {run.name}"
    n = schedule["refusals"]
    verdict = "coverage report" if summary["coverage_report"] else "none"
    cost = (
        "steps not counted"
        if n["steps"] is None
        else f"cost refused {n['fraction_of_steps']:.1%} of {n['steps']} steps"
    )
    yield where, "refusals", f"{n['count']} {json.dumps(dict(n['by_source']))}; {cost}", verdict
    for k in ("extrapolated", "fallback"):
        yield where, f"{k} steps", f"{schedule.get(f'{k}_steps')} of {n['steps']}", "not gated"
    by_method = schedule["seconds_by_method"] or {}
    for g in GROUPS:
        yield where, g, json.dumps(by_method.get(g)), "not gated"
    ratio = c["speed_refused"] or f"{c['speed_ratio']:.1f}x"
    value = f"{ratio}: {c['simulated_seconds']:.0f} simulated s in {c['wall_seconds']:.0f} wall s"
    met = {True: "meets", False: "below"}.get(c["meets_speed_target"], "no result for")
    yield where, "speed", value, f"{met} {c['speed_target_ratio']:g}x"


def rows(cell) -> list[tuple]:
    """The cell's rows, (result, metric, value, verdict), its verdict first."""
    cell = Path(cell)
    real = sorted(r for r in cell.glob("real*") if r.is_dir())
    law = cell / "step_linear"
    if (cell / "law").is_dir():
        raise ValueError(f"{cell} holds law/, a run directory now named step_linear/")
    run = {r.name: json.loads((r / "run.json").read_text()) for r in (*real, law)}
    for name, f in run.items():
        refuse_renamed_keys(f, cell / name / "run.json")
    if len({json.dumps(f.get("workload"), sort_keys=True) for f in run.values()}) != 1:
        raise ValueError(f"{cell}'s runs {sorted(run)} ran different workloads")
    summary = json.loads((law / "summary.json").read_text())
    verdict = "coverage report" if summary["coverage_report"] else "graded"
    out = [("cell", "verdict", f"{len(real)} real runs", verdict)]
    out += _runs(real, law)
    for r in real:
        law_file = run["step_linear"].get("run_file", run["step_linear"])
        laws = entries(law_file, STEP_LINEAR)
        if len(laws) > 1:
            here = keyed_of(json.loads((r / "steps/dp0.engine.json").read_text()))
            laws = [
                _fitted_here(laws, here, lambda e: e.get("engine_config"), STEP_LINEAR)
            ]
        steps, _traces, warmup = runs.graded(r)
        out += [*_warmup(r, warmup), *_cost(r, laws[0], steps)]
        out += _memory(law, r)
    out += [*_warmup(law, runs.graded(law)[2]), *_summary(law, summary)]
    return out


def main(argv=None) -> None:
    print("cell | result | metric | value | verdict")
    for cell in sys.argv[1:] if argv is None else argv:
        for row in rows(cell):
            print(" | ".join((Path(cell).name, *row)))


if __name__ == "__main__":
    main()
