# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The op-level backend: a step priced from the graph captured at its shape.

A step runs at one shape (`step_shape`): a decode step at the CUDA-graph rung
it replayed, or at its width when it replayed none or the backend prices real
rows (``padded=False``, the run file's ``padding: none``), and any other step
as a prefill of its query tokens over its rows. `graph(phase, tokens, seqs)`
gives the region of the `op_graph` captured at that shape, as
`capture.graph.to_json` writes it, or None when there is none.

Each leaf is one pricing unit, `max(t0, F / peak, M / BW(M), t_comm)` over its
work (`op_work.work`): its own tensors at the captured shape, the state it
reads at the step's rows, attention's context in `kv`, the run's KV cache
format. A leaf that launches no kernel costs nothing. Regions compose by the
IR's joins: a `Seq` adds its items in order, a `Repeat` re-emits its body's
prices once per instance, a `max` `Par` costs its longest branch, and a
`resource_bound` `Par` one roofline over its branches' summed work. The price
is one rank's device busy time: the runner takes a DP group's max and
composes host time.

Terms. One per op kind, ``op.<kind>``, and one per join, ``par.<join>``, each
the fold of its prices in run order. A term is `declared`, its note naming
each input, when a spec field the winning bound of one of its prices read is
neither from a datasheet nor probed, and `analytical` otherwise.

Refused: a step with image tokens, since no graph holds the vision encoder; a
two-batch-overlap step, since no graph holds a micro-batch or the MoE
exchanges its comm stream carries; a speculative decode step; a shape with no
graph; and, each reason once, a leaf whose kind has no extractor, whose price
reads a spec field the spec lacks, or whose collective leaves the scale-up
domain.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from atom.compass.backends.base import CostBackend, Granularity
from atom.compass.backends.cost import CostTerm, StepCost, fold_seconds
from atom.compass.backends.op_work import work
from atom.compass.backends.provenance import Method, Provenance, Refusal
from atom.compass.backends.resolver import CostRefused
from atom.compass.backends.shape import IMAGES, BatchView
from atom.compass.backends.step_roofline import _reads
from atom.compass.backends.work import Work, bounds

SOURCE = "op-graph"
TBO = (
    "a two-batch-overlap step: no captured graph holds a micro-batch, nor the "
    "MoE exchanges its comm stream carries"
)
NONE = "; no successor: the analytical law is the last op-level source"


def step_shape(
    rows: Sequence[tuple[int, bool]], rung: int | None
) -> tuple[str, int, int]:
    """The shape a step of `rows`, each its query tokens and whether it
    decodes, runs at, ``(phase, tokens, seqs)``: a decode step at `rung` when it
    replayed one, else at its width, and any other step as a prefill of its
    query tokens over its rows. A speculative decode step is refused."""
    if not all(decode for _, decode in rows):
        return "prefill", sum(q for q, _ in rows), len(rows)
    if wide := [q for q, _ in rows if q != 1]:
        raise ValueError(
            f"decodes {wide[0]} tokens in a row: a speculative decode step is "
            "not captured"
        )
    width = rung or len(rows)
    return "decode", width, width


class OpGraphBackend(CostBackend):
    """A step as its shape's graph, one roofline per leaf, composed by its joins."""

    def __init__(
        self,
        machine,
        graph: Callable[[str, int, int], dict | None],
        *,
        tp: int = 1,
        kv: str | None = None,
        padded: bool = True,
    ) -> None:
        self.machine = machine
        self.graph = graph
        self.tp = tp
        self.kv = kv
        self.padded = padded
        self.name = machine.values.get("name", "")
        self._stated_as: dict[str, str] = {}

    @property
    def granularity(self) -> Granularity:
        return Granularity.OP

    def price(self, batch_view: BatchView) -> StepCost:
        shape, region = self._region(batch_view)
        refused: list[Refusal] = []
        priced = self._priced(region, batch_view, refused)
        if refused:
            raise CostRefused(batch_view, list(dict.fromkeys(refused)))
        seconds: dict[str, list[float]] = {}
        stated: dict[str, list[str]] = {}
        for name, s, notes, _ in priced:
            seconds.setdefault(name, []).append(s)
            stated.setdefault(name, []).extend(notes)
        at = "{} T={} B={}".format(*shape)
        terms = []
        for name, values in seconds.items():
            note = "; ".join(dict.fromkeys(stated[name]))
            method = Method.DECLARED if note else Method.ANALYTICAL
            detail = f"{len(values)} prices of the {at} graph"
            provenance = Provenance(method, detail, SOURCE, note=note and note + NONE)
            terms.append(CostTerm(name, fold_seconds(values), provenance))
        return StepCost(terms)

    def describe(self) -> str:
        rows = "the rows a graph pads to" if self.padded else "real rows"
        return f"op graph at TP {self.tp}, one roofline per leaf, at {rows}"

    def _region(self, view: BatchView) -> tuple[tuple[str, int, int], dict]:
        """The step's shape and the region captured at it."""
        if view.image_tokens:
            reason = IMAGES.format(view.image_tokens)
        elif view.tbo_tokens is not None:
            reason = TBO
        else:
            try:
                rung = view.capture_rung if self.padded else None
                rows = [(r.query_tokens, r.decode) for r in view.requests]
                shape = step_shape(rows, rung)
            except ValueError as exc:
                reason = f"a step that {exc}"
            else:
                region = self.graph(*shape)
                if region is not None:
                    return shape, region
                reason = (
                    "no op_graph holds a {} step of {} tokens over {} sequences; "
                    "compass trace ops captures one".format(*shape)
                )
        raise CostRefused(view, [Refusal(SOURCE, reason)])

    def _priced(self, region: dict, view: BatchView, refused: list) -> list:
        """`region`'s prices in run order, each ``(term, seconds, stated, work)``;
        what refuses goes to `refused`."""
        if "op" in region:
            leaf = self._leaf(region, view, refused)
            return [(f"op.{region['op']}", *leaf)] if leaf else []
        if "repeat" in region:
            return self._priced(region["repeat"], view, refused) * region["count"]
        if "seq" in region:
            return [p for i in region["seq"] for p in self._priced(i, view, refused)]
        branches = [self._priced(b, view, refused) for b in region["par"]]
        summed = sum((p[3] for b in branches for p in b), Work())
        name = f"par.{region['join']}"
        if region["join"] == "max":
            each = [fold_seconds(p[1] for p in b) for b in branches]
            longest = branches[each.index(max(each))]
            return [(name, max(each), [n for p in longest for n in p[2]], summed)]
        unit = self._unit(name, summed, refused)
        return [(name, *unit, summed)] if unit else []

    def _leaf(self, leaf: dict, view: BatchView, refused: list):
        """One leaf's ``(seconds, stated, work)``; None when it launches no
        kernel or refuses."""
        try:
            w = work(leaf, view, self.tp, self.kv)
        except CostRefused as exc:
            refused.extend(exc.declined)
            return None
        except ValueError as exc:
            refused.append(Refusal(SOURCE, f"{leaf['op']}: {exc}"))
            return None
        unit = w and self._unit(leaf["op"], w, refused)
        return unit and (*unit, w)

    def _unit(self, what: str, w: Work, refused: list):
        """One pricing unit's seconds and the stated spec fields its largest
        bound read; None, with its refusal, when it cannot be priced."""
        reads = _reads(w)
        values = self.machine.values
        missing = [p for ps in reads.values() for p in ps if p not in values]
        if missing:
            reason = (
                f"{what} reads {', '.join(dict.fromkeys(missing))}, which the "
                "machine spec does not state"
            )
            refused.append(Refusal(SOURCE, reason))
            return None
        try:
            b = bounds(w, self.machine)
        except ValueError as exc:
            refused.append(Refusal(SOURCE, f"{what}: {exc}"))
            return None
        won = max(b, key=b.get)
        return b[won], [s for p in reads[won] if (s := self._stated(p))]

    def _stated(self, path: str) -> str:
        """`path` and its origin when it is neither a datasheet's nor probed."""
        if path not in self._stated_as:
            origin = self.machine.origin(path)["origin"]
            spec = f"{path} {origin} in the machine spec {self.name}"
            self._stated_as[path] = "" if origin in ("datasheet", "probed") else spec
        return self._stated_as[path]
