#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Step fit's grid points planned from batch shapes, and steps checked for
coverage against a published bundle; `README.md` beside this file says how.

Each step is read where step fit's backend reads it (`StepFitGrid.locate` and
the cell's corners), so the plan and the backend never disagree on a cell.
"""

import argparse
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


def _span(home, nodes, corners, axis: int, margin: int) -> list:
    """`nodes` from the cell's low corner's index on `home` less the margin to
    its high corner's plus the margin."""
    low = home.index(corners[0][0][axis])
    high = home.index(corners[-1][0][axis])
    return nodes[max(0, low - margin) : high + margin + 1]


def _infeasible(target: dict, b: int, n_q: int, c: int, prefill: bool) -> str:
    """Why the target cannot run B rows of N_Q tokens on c cached, or ''."""
    blocks = b * -(-(c + n_q) // target["kv_cache_block_size"])
    checks = [
        (b > target["max_num_seqs"], f"B {b} is above max_num_seqs"),
        (
            prefill and b * n_q > target["max_num_batched_tokens"],
            f"B.N_Q {b * n_q} is above max_num_batched_tokens",
        ),
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


def plan(definition: dict, steps) -> dict:
    """The points to measure for `steps` in the grid `definition` makes."""
    nodes = [*definition["prefill"].values(), definition["decode"]["c"]]
    counts = [definition[k] for k in TARGET[:-1]] + [definition["margin"]]
    counts += definition["capture_sizes"] + [n for axis in nodes for n in axis]
    odd = [n for n in counts if type(n) is not int or n < 0]
    if odd:
        raise SystemExit(f"the grid definition's {odd[0]!r} is not a count")
    ladder = schedulable_ladder(SimpleNamespace(**definition))
    margin = definition["margin"]
    out = {
        "graph_mode": "replay",
        "scope": SCOPE,
        "bounds": {name: definition[name] for name in BOUNDS},
        "prefill": {"nodes": definition["prefill"], "output": [], "no_output": []},
        "decode": {
            "nodes": {
                "B": {str(r): sorted({p + 1, r}) for p, r in zip([0, *ladder], ladder)},
                "c": definition["decode"]["c"],
            },
            "points": [],
        },
    }
    grid = StepFitGrid({**out, "unit": "s"})
    planned, unplaced = defaultdict(set), []
    for where, view in steps:
        try:
            cells, at = grid.locate(view)
            corners = cells.corners(view, at)
        except CostRefused as refused:
            unplaced.append({"step": where, "reason": _reason(refused)})
            continue
        if view.prefill:
            spans = [_span(n, n, corners, i, margin) for i, n in enumerate(cells.nodes)]
            key = "output" if view.produces_output else "no_output"
            planned["prefill", key].update(itertools.product(*spans))
            continue
        k = ladder.index(view.capture_rung)
        for rung in ladder[max(0, k - margin) : k + margin + 1]:
            near = grid.decode[rung].nodes
            spans = [
                _span(h, n, corners, i, margin)
                for i, (h, n) in enumerate(zip(cells.nodes, near))
            ]
            planned["decode", "points"].update(
                (rung, *p) for p in itertools.product(*spans)
            )
    # A bundle whose measured rungs skip one of the ladder's does not load.
    decode = planned["decode", "points"]
    rungs = sorted({p[0] for p in decode})
    c_nodes = sorted({p[2] for p in decode})
    for rung in ladder:
        if rungs and rungs[0] < rung < rungs[-1] and rung not in rungs:
            decode.update(
                itertools.product([rung], grid.decode[rung].nodes[0], c_nodes)
            )
    out |= {"counts": {}, "infeasible": [], "unplaced": unplaced}
    for (phase, key), points in sorted(planned.items()):
        for point in sorted(points):
            at = [int(x) for x in point]
            if phase == "prefill":
                name, why = f"prefill {key}", _infeasible(definition, *at, prefill=True)
            else:
                name = f"decode rung {at[0]}"
                why = _infeasible(definition, at[1], 1, at[2], prefill=False)
            if why:
                out["infeasible"].append({"grid": name, "at": at, "reason": why})
                continue
            out[phase][key].append(at)
            out["counts"][name] = out["counts"].get(name, 0) + 1
    return out


def coverage(backend, steps) -> dict:
    """Per step, whether every corner of its cell was measured and the
    residual answers it, over the grid and residual of `backend`."""
    covered, missing, refused = 0, defaultdict(set), []
    for where, view in steps:
        gaps, reasons = (), []
        try:
            cells, at = backend.grid.locate(view)
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
