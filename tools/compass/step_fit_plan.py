#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Step fit's grid points planned from batch shapes, and steps checked for
coverage against a published bundle; `README.md` beside this file says how.

Each step is read where step fit's backend reads it (`StepFitGrid.locate` and
the cell's corners), so the plan and the backend never disagree on a cell.
"""

import argparse
import bisect
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from atom.compass import parity
from atom.compass.backends import BatchView, CostRefused, RequestShape, StepFitGrid
from atom.compass.run import _width_keys
from atom.compass.run.step_fit import BOUNDS, SCOPE, load
from atom.compass.runner.overrides import schedulable_ladder
from atom.compass.spec import MachineSpec

#: The target a grid definition states beside its nodes and margin.
TARGET = (*BOUNDS, "kv_cache_block_size", "num_kvcache_blocks", "capture_sizes")
#: The resolved ATOM config fields `load` reads, by their `Config` names, with
#: ``data_parallel_size`` from its parallel config and ``quantization`` the
#: hash of its quantization config.
CONFIG = (
    "model",
    "tensor_parallel_size",
    "data_parallel_size",
    "enforce_eager",
    "capture_sizes",
    *BOUNDS,
    "kv_cache_block_size",
    "kv_cache_dtype",
    "quantization",
)


def _read(path: str, keys) -> dict:
    mapping = json.loads(Path(path).read_text())
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise SystemExit(f"{path} states no {', '.join(missing)}")
    return mapping


def read_steps(records) -> list[tuple[str, BatchView]]:
    """Each recorded step, named, as the view a simulated runner prices."""
    read = [
        (
            f"{record} dp{dp} step {line['step']}",
            BatchView(
                tuple(RequestShape(*row) for row in line["rows"]),
                capture_rung=line["rung"],
                produces_output=line["produces_output"],
            ),
        )
        for record in records
        for dp, lines in parity.read(record).items()
        for line in lines
    ]
    if not read:
        raise SystemExit(f"{', '.join(map(str, records))} record no step")
    return read


def _reason(refused: CostRefused) -> str:
    return "; ".join(r.reason for r in refused.declined)


def _span(nodes, low, high, margin: int) -> list:
    """`nodes` from the last at or below `low` to the first at or above `high`,
    widened by `margin` nodes either way."""
    first = bisect.bisect_right(nodes, low) - 1
    last = bisect.bisect_left(nodes, high)
    return nodes[max(0, first - margin) : last + margin + 1]


def _infeasible(target: dict, b: int, t: int, c: int, prefill: bool) -> str:
    """Why the target cannot run B rows of T tokens in all on c cached each,
    or ''."""
    n_q = t // b
    blocks = b * -(-(c + n_q) // target["kv_cache_block_size"])
    checks = [
        (b > target["max_num_seqs"], f"B {b} is above max_num_seqs"),
        (
            prefill and t > target["max_num_batched_tokens"],
            f"T {t} is above max_num_batched_tokens",
        ),
        (t < b, f"T {t} is below B {b}, under one token a row"),
        (t % b, f"N_Q = T/B = {t}/{b} is not an integer"),
        (
            c + n_q > target["max_model_len"],
            f"c + N_Q {c + n_q} is above max_model_len",
        ),
        (
            blocks > target["num_kvcache_blocks"],
            f"{blocks} KV blocks are above num_kvcache_blocks",
        ),
    ]
    return next((why for over, why in checks if over), "")


def _c_max(target: dict, b: int) -> int:
    """The largest cached context at which B decode rows fit the KV pool."""
    return target["kv_cache_block_size"] * (target["num_kvcache_blocks"] // b) - 1


def _capped(nodes, top: int) -> list:
    """`nodes` below `top`, then `top`."""
    return sorted({n for n in nodes if n < top} | {top})


def _layout(definition: dict, b_nodes: dict) -> dict:
    """The grid `definition` makes with these decode B nodes per rung, its
    nodes past a bound dropped and the bounds added: the token budget T's top,
    ``max_num_seqs`` B's and ``max_model_len`` less the budget c's for
    prefill; per decode rung, each B node's KV limit and ``max_model_len - 1``
    as c nodes."""
    budget = definition["max_num_batched_tokens"]
    tops = {
        "B": definition["max_num_seqs"],
        "T": budget,
        "c": definition["max_model_len"] - budget,
    }
    c, top = definition["decode"]["c"], definition["max_model_len"] - 1
    return {
        "graph_mode": "replay",
        "scope": SCOPE,
        "bounds": {name: definition[name] for name in BOUNDS},
        "prefill": {
            "nodes": {a: _capped(definition["prefill"][a], t) for a, t in tops.items()},
            "output": [],
            "no_output": [],
        },
        "decode": {
            "nodes": {
                "B": {str(r): sorted(bs) for r, bs in b_nodes.items()},
                "c": {
                    str(r): _capped(c + [_c_max(definition, b) for b in bs], top)
                    for r, bs in b_nodes.items()
                },
            },
            "points": [],
        },
    }


def plan(definition: dict, steps) -> dict:
    """The points to measure for `steps` in the grid `definition` makes."""
    if sorted(definition["prefill"]) != ["B", "T", "c"]:
        raise SystemExit(
            f"the grid definition's prefill nodes are on "
            f"{', '.join(definition['prefill'])}, not B, T, c"
        )
    nodes = [*definition["prefill"].values(), definition["decode"]["c"]]
    counts = [definition[k] for k in TARGET[:-1]] + [definition["margin"]]
    counts += definition["capture_sizes"] + [n for axis in nodes for n in axis]
    odd = [n for n in counts if type(n) is not int or n < 0]
    if odd:
        raise SystemExit(f"the grid definition's {odd[0]!r} is not a count")
    if definition["max_model_len"] < definition["max_num_batched_tokens"]:
        raise SystemExit(
            "the grid definition's max_model_len is below its "
            "max_num_batched_tokens, so prefill's top c node, max_model_len "
            "less the token budget, would be negative"
        )
    ladder = schedulable_ladder(SimpleNamespace(**definition))
    margin = definition["margin"]
    b_nodes = {r: {p + 1, r} for p, r in zip([0, *ladder], ladder)}
    # A B at which a recorded step reaches past its rung's KV limit is a B node
    # of the rung, so that its own limit is a c node there.
    grid = StepFitGrid({**_layout(definition, b_nodes), "unit": "s"})
    for _, view in steps:
        if view.prefill or view.capture_rung not in b_nodes:
            continue
        try:
            _, (b, c) = grid.locate(view)
        except CostRefused:
            continue
        if c > _c_max(definition, view.capture_rung):
            b_nodes[view.capture_rung].add(int(b))
    out = _layout(definition, b_nodes)
    grid = StepFitGrid({**out, "unit": "s"})
    planned, unplaced = defaultdict(set), []
    for where, view in steps:
        try:
            cells, at = grid.locate(view)
            corners = cells.corners(view, at)
        except CostRefused as refused:
            unplaced.append({"step": where, "reason": _reason(refused)})
            continue
        low, high = corners[0][0], corners[-1][0]
        if view.prefill:
            spans = [
                _span(n, low[i], high[i], margin) for i, n in enumerate(cells.nodes)
            ]
            key = "output" if view.produces_output else "no_output"
            planned["prefill", key].update(itertools.product(*spans))
            continue
        # On a rung either side, every B node, at the c nodes spanning the cell's.
        k = ladder.index(view.capture_rung)
        for rung in ladder[max(0, k - margin) : k + margin + 1]:
            bs, cs = grid.decode[rung].nodes
            if rung == view.capture_rung:
                bs = _span(bs, low[0], high[0], margin)
            planned["decode", "points"].update(
                (rung, *p)
                for p in itertools.product(bs, _span(cs, low[1], high[1], margin))
            )
    # A bundle whose measured rungs skip one of the ladder's does not load.
    decode = planned["decode", "points"]
    rungs = sorted({p[0] for p in decode})
    c_planned = [p[2] for p in decode]
    for rung in ladder:
        if rungs and rungs[0] < rung < rungs[-1] and rung not in rungs:
            bs, cs = grid.decode[rung].nodes
            spanned = _span(cs, min(c_planned), max(c_planned), 0)
            decode.update(itertools.product([rung], bs, spanned))
    out |= {"counts": {}, "infeasible": [], "unplaced": unplaced}
    for (phase, key), points in sorted(planned.items()):
        for point in sorted(points):
            at = [int(x) for x in point]
            if phase == "prefill":
                name, why = f"prefill {key}", _infeasible(definition, *at, prefill=True)
            else:
                name = f"decode rung {at[0]}"
                why = _infeasible(definition, at[1], at[1], at[2], prefill=False)
            if why:
                out["infeasible"].append({"grid": name, "at": at, "reason": why})
                continue
            out[phase][key].append(at)
            out["counts"][name] = out["counts"].get(name, 0) + 1
    return out


def coverage(backend, steps) -> dict:
    """Per step, whether every corner of its cell was measured and the
    residual answers it, over the grid and residual of `backend`; and, apart,
    how many prefill steps are cached past the top c node."""
    covered, missing, refused, above = 0, defaultdict(set), [], 0
    for where, view in steps:
        gaps, reasons = (), []
        try:
            cells, at = backend.grid.locate(view)
            above += bool(view.prefill) and at[-1] > cells.nodes[-1][-1]
            corners = cells.corners(view, at)
            gaps = {c for c, _ in corners if c not in cells.points}
            missing[cells.name] |= gaps
        except CostRefused as r:
            reasons.append(_reason(r))
        try:
            backend.residual.factor(view)
        except CostRefused as r:
            reasons.append(_reason(r))
        if reasons:
            refused.append({"step": where, "reasons": reasons})
        covered += not (gaps or reasons)
    return {
        "steps": len(steps),
        "covered": covered,
        "share": covered / len(steps),
        "missing": {
            name: sorted([int(x) for x in corner] for corner in corners)
            for name, corners in sorted(missing.items())
            if corners
        },
        "refused": refused,
        "prefill_above_top_c": above,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="step_fit_plan.py", description=__doc__.splitlines()[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)
    planning = commands.add_parser("plan", help="the points to measure")
    planning.add_argument("definition", help="the grid definition, JSON")
    checking = commands.add_parser("coverage", help="the steps a bundle covers")
    checking.add_argument("run", help="the run file naming the bundle")
    checking.add_argument("config", help="the run's resolved ATOM config, JSON")
    for command in (planning, checking):
        command.add_argument("records", nargs="+", help="step record directories")
    args = parser.parse_args(argv)
    read = read_steps(args.records)
    if args.command == "plan":
        definition = _read(args.definition, ("prefill", "decode", "margin", *TARGET))
        print(json.dumps(plan(definition, read), indent=1))
        return 0
    run, fields = _read(args.run, ("step_fit", "machine")), _read(args.config, CONFIG)
    config = SimpleNamespace(
        **fields,
        parallel_config=SimpleNamespace(
            data_parallel_size=fields["data_parallel_size"]
        ),
        quant_config=SimpleNamespace(compute_hash=lambda: fields["quantization"]),
    )
    machine = MachineSpec.from_mapping(_width_keys(run["machine"]))
    backend, _ = load(run["step_fit"], config, machine)
    report = coverage(backend, read)
    print(json.dumps(report, indent=1))
    return 0 if report["covered"] == report["steps"] else 1


if __name__ == "__main__":
    sys.exit(main())
