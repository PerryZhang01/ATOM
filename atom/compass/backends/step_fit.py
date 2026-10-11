# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Tier a: a step looked up in a grid of measured whole forwards.

A batch maps to the homogeneous shape that keeps its token count and its
attention work, Sum (N_Q.c + N_Q(N_Q+1)/2) over rows with c the cached
context, in exact arithmetic: B rows of N_Q = tokens / B each, and the c that
gives the same work. That shape is read in one grid:

    prefill:                (B, N_Q, c), from the grid of chunks that produce
                            output or the grid of chunks that do not
    decode, replay bundle:  (B, c), from the grid of the rung the step replays
    decode, eager bundle:   (B, c), the rung ignored

A cell is the box between adjacent nodes on every axis; a coordinate on a node
reads that node alone, so an exact hit needs one point. The answer is the
multilinear interpolation over the cell's corners when every corner was
measured, and otherwise the step is refused naming the missing corners. Nothing
outside a cell answers: no extrapolation, clamp or fallback. A missing point
does not fail loading; it refuses the steps that need it.

Also refused: a step mixing prefill and decode rows, a decode row computing
more than one token, a count that is not an int, a prefill step that does not
say whether it produces output, and under a replay bundle a decode step that
replays no graph, being wider than the widest rung or eager.

The answer is one term for the whole forward, never multiplied by the TP
width, since the measurement already holds the collectives. Its species is
`measured` only for a homogeneous batch on a node, `interpolated` otherwise.

The grid is a mapping, as a bundle holds it:

    {"graph_mode": "replay" | "eager", "scope": <timing scope>,
     "unit": "s" | "ms" | "us",
     "prefill": {"nodes": {"B": [...], "N_Q": [...], "c": [...]},
                 "output": [POINT, ...], "no_output": [POINT, ...]},
     "decode": {"nodes": {"B": B_NODES, "c": [...]}, "points": [POINT, ...]}}

    replay: B_NODES = {"<rung>": [B, ...], ...}, a POINT at [rung, B, c]
    eager:  B_NODES = [B, ...],                  a POINT at [B, c]
    POINT = {"at": [...], "time": <in unit>, "statistic": <name>, "repeats": n}
"""

from __future__ import annotations

import bisect
import itertools
import math
from collections.abc import Mapping, Sequence
from fractions import Fraction

from atom.compass.backends.base import CostBackend, Tier
from atom.compass.backends.cost import CostTerm, StepCost
from atom.compass.backends.ladder import CostRefused
from atom.compass.backends.provenance import Provenance, Refusal, Species
from atom.compass.backends.shape import (
    BatchView,
    RequestShape,
    sum_query_cached,
    sum_query_square,
    sum_tokens,
)

SOURCE = "step-fit"
UNITS = {"s": Fraction(1), "ms": Fraction(1, 10**3), "us": Fraction(1, 10**6)}


def equivalent_shape(
    rows: Sequence[RequestShape],
) -> tuple[Fraction, Fraction, Fraction]:
    """(B, N_Q, c) of the homogeneous batch with these rows' tokens and work."""
    rows_n, tokens = len(rows), sum_tokens(rows)
    # Sum N_Q(N_Q+1)/2 is (Sum N_Q^2 + Sum N_Q) / 2, and N_Q^2 + N_Q is even.
    work = sum_query_cached(rows) + (sum_query_square(rows) + tokens) // 2
    query = Fraction(tokens, rows_n)
    return (
        Fraction(rows_n),
        query,
        (Fraction(work, rows_n) - query * (query + 1) / 2) / query,
    )


def _refused(view: BatchView, reason: str) -> CostRefused:
    return CostRefused(view, [Refusal(SOURCE, reason)])


def _named(axes: Sequence[str], at: Sequence[Fraction]) -> str:
    return ", ".join(f"{axis}={value}" for axis, value in zip(axes, at))


class _Cells:
    """One grid: ascending nodes per axis, and the points measured on them."""

    def __init__(self, name: str, axes: tuple[str, ...], nodes, points) -> None:
        self.name, self.axes = name, axes
        self.nodes = [[Fraction(n) for n in axis_nodes] for axis_nodes in nodes]
        for axis, axis_nodes in zip(axes, self.nodes):
            if not axis_nodes or any(a >= b for a, b in itertools.pairwise(axis_nodes)):
                raise ValueError(
                    f"the {name} grid's {axis} nodes "
                    f"[{', '.join(map(str, axis_nodes))}] are not strictly ascending"
                )
        self.points: dict[tuple[Fraction, ...], Fraction] = {}
        for point in points:
            at = tuple(Fraction(v) for v in point["at"])
            time = float(point["time"])
            repeats = point["repeats"]
            if len(at) != len(axes) or not all(v in n for v, n in zip(at, self.nodes)):
                raise ValueError(
                    f"a {name} point at {point['at']} is not on the grid's "
                    f"{'/'.join(axes)} nodes"
                )
            if at in self.points:
                raise ValueError(f"the {name} grid measures {point['at']} twice")
            if not (math.isfinite(time) and time > 0.0):
                raise ValueError(
                    f"the {name} point at {point['at']} takes {time}, which is "
                    "not a positive duration"
                )
            if not point["statistic"] or type(repeats) is not int or repeats < 1:
                raise ValueError(
                    f"the {name} point at {point['at']} records no statistic "
                    "and repeat count"
                )
            self.points[at] = Fraction(time)

    def lookup(
        self, view: BatchView, at: tuple[Fraction, ...]
    ) -> tuple[Fraction, bool, str]:
        """The value at `at`, whether it is a node, and the cell; or a refusal."""
        sides = []
        for axis, nodes, x in zip(self.axes, self.nodes, at):
            if x in nodes:
                sides.append(((x, Fraction(1)),))
                continue
            i = bisect.bisect(nodes, x)
            if i in (0, len(nodes)):
                raise _refused(
                    view,
                    f"{axis}={x} is outside the {self.name} grid's nodes "
                    f"{nodes[0]}..{nodes[-1]}",
                )
            low, high = nodes[i - 1], nodes[i]
            weight = (x - low) / (high - low)
            sides.append(((low, 1 - weight), (high, weight)))
        corners = [
            (tuple(n for n, _ in c), math.prod(w for _, w in c))
            for c in itertools.product(*sides)
        ]
        missing = [corner for corner, _ in corners if corner not in self.points]
        if missing:
            raise _refused(
                view,
                f"the {self.name} cell holding {_named(self.axes, at)} has no "
                f"measurement at {'; '.join(_named(self.axes, m) for m in missing)}",
            )
        value = sum(self.points[corner] * weight for corner, weight in corners)
        cell = ", ".join(
            f"{axis}={s[0][0]}" if len(s) == 1 else f"{axis} {s[0][0]}..{s[1][0]}"
            for axis, s in zip(self.axes, sides)
        )
        return value, len(corners) == 1, cell


class StepFitGrid:
    """A bundle's grids: prefill by whether a chunk produces output, and decode."""

    def __init__(self, bundle: Mapping) -> None:
        self.graph_mode = bundle["graph_mode"]
        if self.graph_mode not in ("replay", "eager"):
            raise ValueError(
                f"graph mode {self.graph_mode!r} is neither replay nor eager"
            )
        if bundle["unit"] not in UNITS:
            raise ValueError(f"unit {bundle['unit']!r} is not one of {list(UNITS)}")
        self.unit = UNITS[bundle["unit"]]
        self.scope = bundle["scope"]
        prefill = bundle["prefill"]
        axes = ("B", "N_Q", "c")
        nodes = [prefill["nodes"][axis] for axis in axes]
        self.prefill = {
            produces: _Cells(f"prefill {key}", axes, nodes, prefill[key])
            for produces, key in ((True, "output"), (False, "no_output"))
        }
        decode = bundle["decode"]
        b_nodes, c_nodes = decode["nodes"]["B"], decode["nodes"]["c"]
        replay = self.graph_mode == "replay"
        if isinstance(b_nodes, Mapping) != replay:
            raise ValueError(
                f"under graph mode {self.graph_mode} the decode B nodes are "
                f"{'one list per rung' if replay else 'one list'}, not a "
                f"{type(b_nodes).__name__}"
            )
        if not replay:
            self.decode = _Cells(
                "decode", ("B", "c"), [b_nodes, c_nodes], decode["points"]
            )
            return
        by_rung = {int(rung): [] for rung in b_nodes}
        for point in decode["points"]:
            rung, *at = point["at"]
            if rung not in by_rung:
                raise ValueError(
                    f"a decode point at {point['at']} is on rung {rung}, which "
                    "has no B nodes"
                )
            by_rung[rung].append({**point, "at": at})
        self.decode = {
            int(rung): _Cells(
                f"decode rung {rung}", ("B", "c"), [nodes, c_nodes], by_rung[int(rung)]
            )
            for rung, nodes in b_nodes.items()
        }


class StepFitBackend(CostBackend):
    """Prices a step as the grid's value at the batch's equivalent shape."""

    def __init__(self, grid: StepFitGrid) -> None:
        self.grid = grid

    @property
    def tier(self) -> Tier:
        return Tier.COARSE

    def estimate(self, batch_view: BatchView) -> StepCost:
        if not isinstance(batch_view, BatchView):
            raise TypeError(
                f"this backend prices a BatchView, not {type(batch_view).__name__}"
            )
        view, rows = batch_view, batch_view.requests
        prefill, decode = view.prefill, view.decode
        if prefill and decode:
            raise _refused(
                view,
                f"a step of {len(prefill)} prefill and {len(decode)} decode rows; "
                "the grid prices a step of one kind",
            )
        counts = [n for row in rows for n in (row.query_tokens, row.context_tokens)]
        odd = [
            n
            for n in counts + [view.capture_rung]
            if n is not None and type(n) is not int
        ]
        if odd:
            raise _refused(view, f"the count {odd[0]!r} is not an int")
        rows_n, query, cached = equivalent_shape(rows)
        if prefill:
            if view.produces_output is None:
                raise _refused(
                    view,
                    "a prefill step that does not say whether it produces "
                    "output, which picks its grid",
                )
            cells, at = self.grid.prefill[view.produces_output], (rows_n, query, cached)
        else:
            widest = max(row.query_tokens for row in decode)
            if widest > 1:
                raise _refused(
                    view,
                    f"a decode row computes {widest} tokens; the decode grid has "
                    "no N_Q axis",
                )
            at = (rows_n, cached)
            if self.grid.graph_mode == "eager":
                cells = self.grid.decode
            elif view.capture_rung is None:
                raise _refused(
                    view,
                    f"a decode step of {len(decode)} rows replays no graph, "
                    "and a replay bundle has no eager decode",
                )
            else:
                cells = self.grid.decode.get(view.capture_rung)
                if cells is None:
                    raise _refused(view, f"no decode rung {view.capture_rung}")
        value, on_node, cell = cells.lookup(view, at)
        homogeneous = len({(r.query_tokens, r.context_tokens) for r in rows}) == 1
        species = Species.MEASURED if on_node and homogeneous else Species.INTERPOLATED
        seconds = float(value * self.grid.unit)
        detail = f"lookup {seconds!r} s, {species}; {cells.name} cell {cell}"
        return StepCost(
            [CostTerm("forward", seconds, Provenance(species, detail, SOURCE))]
        )

    def describe(self) -> str:
        return (
            f"tier-a step fit; {self.grid.graph_mode} decode grid, "
            f"timing scope {self.grid.scope}"
        )
