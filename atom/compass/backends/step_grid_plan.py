# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The step grid's points planned from batch shapes.

`plan` takes steps as step-record lines (`atom/compass/parity/`), each in a
`(name, line)` pair whose name a report gives the step, and reads each step
where the step grid backend reads it (`StepGrid.locate` and the cell's
corners), so the plan and the backend never disagree on a cell. Only shapes
are read, never timestamps. Reading records stays with the caller.

`plan` reads its target from the engine its inputs' ranks state, each in its
`dp<r>.engine.json`: `TARGET`, the scheduler's bounds, the KV block size and
count, and the decode ladder. Inputs whose ranks state two engines are
refused, and so is a step whose decode row computes more than one token: its
run had a drafter, and the ladder its engine states holds rungs ATOM did not
capture. The grid definition is a mapping:

    {"prefill": {"B": [...], "T": [...], "c": [...]}, "decode": {"c": [...]},
     "margin": n}

`T` is a step's token count and `c` its cached context; `margin` is the cells
added outward on every axis, and the rungs either side on the ladder. Every
number is a non-negative int. The definition may also state a `TARGET` field,
which is refused unless the engine states the same value.

`plan` first places the nodes the target's bounds give, dropping any
definition node past one. Prefill's top nodes are `max_num_seqs` on B,
`max_num_batched_tokens` on T and `max_model_len - max_num_batched_tokens` on
c, so `max_model_len` below the token budget is refused. Each decode rung has
B nodes at the previous rung + 1, at the rung, and at each B between them that
a step on the rung reaches past the rung's KV limit; its c nodes add each B
node's KV limit, `block size * (num_kvcache_blocks // B) - 1`, and
`max_model_len - 1`.

It then plans every corner of each step's cell, widened by `margin` nodes on
every axis and to `margin` rungs either side, each such rung at all its B
nodes; and, on any rung between two planned rungs, its B nodes at its c nodes
spanning those planned, since a bundle whose measured rungs skip one of the
ladder's does not load. A point that is no batch the target runs is not
planned: B above `max_num_seqs`, T above `max_num_batched_tokens`, T not a
multiple of B or below it, c + N_Q above `max_model_len`, or
B * ceil((c + N_Q) / block size) above the block count, assuming no prefix
sharing.
"""

import bisect
import itertools
from collections import defaultdict

from atom.compass.backends.resolver import CostRefused
from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.backends.step_grid import StepGrid

#: The one timing scope a bundle may declare, the whole `ModelRunner.forward`.
SCOPE = "forward"
#: The scheduler's bounds a bundle records, which its grid was planned for.
BOUNDS = ("max_num_seqs", "max_num_batched_tokens", "max_model_len")
#: The engine fields `plan` plans for.
TARGET = (*BOUNDS, "kv_cache_block_size", "num_kvcache_blocks", "ladder")
#: The grid definition's own fields.
DEFINITION = ("prefill", "decode", "margin")


def _views(steps) -> list[tuple[str, BatchView]]:
    """Each named step-record line as the view a simulated runner prices."""
    views = [
        (
            name,
            BatchView(
                tuple(RequestShape(*row) for row in line["rows"]),
                capture_rung=line["rung"],
                produces_output=line["produces_output"],
            ),
        )
        for name, line in steps
    ]
    if not views:
        raise ValueError("there is no step to read")
    return views


def _target(definition: dict, engines: dict) -> dict:
    """The `TARGET` fields of the one engine every input rank states."""
    if not engines:
        raise ValueError("no input states the engine its steps ran on")
    (first, engine), *rest = engines.items()
    for name, other in rest:
        keys = sorted(engine.keys() | other.keys())
        differ = [k for k in keys if engine.get(k) != other.get(k)]
        if differ:
            raise ValueError(
                f"{first} and {name} state two engines, which differ on "
                f"{', '.join(differ)}"
            )
    missing = [key for key in TARGET if key not in engine]
    if missing:
        raise ValueError(f"{first} states no {', '.join(missing)}")
    unknown = sorted(set(definition) - {*DEFINITION, *TARGET})
    if unknown:
        raise ValueError(
            f"the grid definition states {', '.join(unknown)}, which plan does "
            "not read; its target is the engine's"
        )
    missing = [key for key in DEFINITION if key not in definition]
    if missing:
        raise ValueError(f"the grid definition states no {', '.join(missing)}")
    differ = [k for k in TARGET if k in definition and definition[k] != engine[k]]
    if differ:
        raise ValueError(
            "the grid definition states "
            + "; ".join(
                f"{k} {definition[k]!r} and the engine {engine[k]!r}" for k in differ
            )
        )
    return {key: engine[key] for key in TARGET}


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


def _line(step: int, phase: str, key: str, at: list) -> dict:
    """The planned point `at`, `[B, T, c]` or `[rung, B, c]`, as a step-record
    line: B rows of N_Q = T/B tokens, or of one decode token, on c cached."""
    if phase == "prefill":
        (b, t, c), rung, output = at, None, key == "output"
        row = [t // b, c + t // b, False]
    else:
        (rung, b, c), output = at, True
        row = [1, c + 1, True]
    return {
        "step": step,
        "dp_rank": 0,
        "is_dummy_run": False,
        "produces_output": output,
        "batch": [],
        "named": [],
        "rows": [list(row) for _ in range(b)],
        "rung": rung,
        "origin": "plan",
    }


def plan(definition: dict, steps, engines: dict) -> dict:
    """The points to measure for `steps` in the grid `definition` makes for
    the engine `engines` states, a mapping from each input rank's name to the
    engine it states.

    It returns `graph_mode`, `scope` and `bounds`, and `prefill` and `decode`
    in a bundle grid's layout (`step_grid.py`) with no point measured;
    `lines`, each planned point as a step-record line with origin `plan`,
    which that grid locates on the point's node; `counts` per grid;
    `infeasible`, the points not planned and why; and `unplaced`, the steps
    outside the grid with the backend's reason.
    """
    steps = _views(steps)
    definition = {**definition, **_target(definition, engines)}
    drafted = [n for n, view in steps if any(r.query_tokens > 1 for r in view.decode)]
    if drafted:
        raise ValueError(
            f"{drafted[0]} decodes more than one token in a row, so its run had "
            "a drafter, and the ladder its engine states holds rungs ATOM did "
            "not capture"
        )
    if sorted(definition["prefill"]) != ["B", "T", "c"]:
        raise ValueError(
            f"the grid definition's prefill nodes are on "
            f"{', '.join(definition['prefill'])}, not B, T, c"
        )
    ladder = definition["ladder"]
    nodes = [*definition["prefill"].values(), definition["decode"]["c"]]
    counts = [definition[k] for k in TARGET[:-1]] + [definition["margin"]]
    counts += ladder + [n for axis in nodes for n in axis]
    odd = [n for n in counts if type(n) is not int or n < 0]
    if odd:
        raise ValueError(f"the grid definition's or engine's {odd[0]!r} is not a count")
    if definition["max_model_len"] < definition["max_num_batched_tokens"]:
        raise ValueError(
            "the engine's max_model_len is below its max_num_batched_tokens, so "
            "prefill's top c node, max_model_len less the token budget, would "
            "be negative"
        )
    margin = definition["margin"]
    b_nodes = {r: {p + 1, r} for p, r in zip([0, *ladder], ladder)}
    # A B at which a step reaches past its rung's KV limit is a B node of the
    # rung, so that its own limit is a c node there.
    grid = StepGrid({**_layout(definition, b_nodes), "unit": "s"})
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
    grid = StepGrid({**out, "unit": "s"})
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
    out |= {"lines": [], "counts": {}, "infeasible": [], "unplaced": unplaced}
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
            out["lines"].append(_line(len(out["lines"]), phase, key, at))
            out["counts"][name] = out["counts"].get(name, 0) + 1
    return out
