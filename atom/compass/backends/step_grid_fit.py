# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The step grid's bundle, from its grid's measured points and its residual's targets.

`bundle(definition, planned, points, targets, seed)` takes the grid definition,
`step_grid_plan.plan`'s output for it, each planned line's measurement in the
plan's line order (a mapping holding `seconds`, `statistic` and `repeats`, the
kept repeats), and the residual's targets, `step_labels.Sample`s whose
provenance names their `source`. Each measurement becomes the grid point on its
line's node, with its statistic and the count of its kept repeats: prefill's
`[B, T, c]` in the grid of chunks that produce output or of those that do not,
decode's `[rung, B, c]`. The grid keeps the plan's nodes, graph mode and timing
scope, holds seconds, and states `VALUE`, what its points hold.

A target with a device label that the grid answers becomes a triple of its
view, the lookup and its label, and `step_grid_residual.fit` fits the residual
on the triples. A target the grid refuses is named in the report with the
grid's reason; one with no device label is counted by why it was cut. A fit
with no target the grid answers refuses.

The bundle is the member `run/step_grid.py` loads: the grid, the residual, the
plan's bounds, the definition, and the sources its triples came from. The
report is the residual's by kind, each kind's selected member also broken down
by source: the triples fitted and held out, the median and scaled MAD of its
relative residuals over those fitted, and the median absolute one over those
held out, a band it holds no factor for counting as a factor of 0, as the fit
counts it.
"""

import copy
import statistics
from collections import Counter

from atom.compass.backends.resolver import CostRefused
from atom.compass.backends.step_grid import StepGrid
from atom.compass.backends.step_grid_residual import RECIPE, Residual, fit, kind

#: What a bundle's point holds: `measure steps`' bench window, the device time
#: from the forward's entry to its exit with work queued ahead.
VALUE = "bench window"


def bundle(definition, planned, points, targets, seed: int) -> tuple[dict, dict]:
    """The bundle and its report; or a ValueError."""
    keys = ("graph_mode", "scope", "prefill", "decode")
    grid = copy.deepcopy({k: planned[k] for k in keys}) | {"unit": "s", "value": VALUE}
    for line, measurement in zip(planned["lines"], points, strict=True):
        (n_q, context, decode), b = line["rows"][0], len(line["rows"])
        point = {"statistic": measurement["statistic"]}
        point["repeats"] = len(measurement["repeats"])
        point["time"] = measurement["seconds"]
        if decode:
            at, into = [line["rung"], b, context - 1], grid["decode"]["points"]
        else:
            at = [b, b * n_q, context - n_q]
            into = grid["prefill"]["output" if line["produces_output"] else "no_output"]
        into.append({"at": at, **point})
    lookup = StepGrid(grid)
    triples, sources, unpriced, cut = [], [], [], Counter()
    for s in targets:
        if s.device_s is None:
            cut[s.cut] += 1
            continue
        try:
            cells, at = lookup.locate(s.view)
            value = cells.lookup(s.view, at)[0]
        except CostRefused as refused:
            reason = "; ".join(r.reason for r in refused.declined)
            run = s.provenance["run"]
            unpriced.append(
                {"step": f"{run} dp{s.rank} step {s.step}", "reason": reason}
            )
            continue
        triples.append((s.view, float(value * lookup.unit), s.device_s))
        sources.append(s.provenance["source"])
    if not triples:
        raise ValueError(
            f"none of the {len(targets)} targets is a step the grid answers, "
            "with a device label and inside a complete cell"
        )
    residual, kinds = fit(triples, seed)
    fitted = Residual(residual).kinds
    for k, report in kinds.items():
        held, by = set(report["held_out"]), {}
        for i, (view, looked_up, measured) in enumerate(triples):
            if kind(view) != k:
                continue
            x = tuple(float(v) for v in RECIPE(k, view.requests))
            try:
                factor = sum(w * m(view, x) for _, w, m in fitted[k].members)
            except LookupError:
                factor = 0.0
            errors = by.setdefault(sources[i], ([], []))
            errors[i in held].append(factor * looked_up / measured - 1.0)
        report["sources"] = {source: _spread(*e) for source, e in sorted(by.items())}
    out = {"grid": grid, "residual": residual, "bounds": planned["bounds"]}
    out |= {"definition": definition, "sources": sorted(set(sources))}
    return out, {"kinds": kinds, "unpriced": unpriced, "cut": dict(cut)}


def _spread(fitted: list, held: list) -> dict:
    centre = statistics.median(fitted) if fitted else None
    return {
        "fitted": len(fitted),
        "held_out": len(held),
        "centre": centre,
        "mad": (
            1.4826 * statistics.median(abs(e - centre) for e in fitted)
            if fitted
            else None
        ),
        "held_out_error": statistics.median(map(abs, held)) if held else None,
    }
