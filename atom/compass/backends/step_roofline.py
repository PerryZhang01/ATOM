# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The step roofline, and the step-level fallback to it.

`StepRooflineBackend` prices a step from `Module`s, plain descriptors of one
rank's modules that `atom.compass.run.modules.walk` reads off the meta build.
Each module's work comes from its per-rank shapes and the batch (`work`), and
the module costs one roofline, `max(t0, F / peak, M / BW(M))`, plus the
collective it joins; a module whose collective runs inside its own kernel
(`fused`) takes the collective inside that max instead. The step is the sum
over modules, device busy time only: host time is the runner's host
recurrence, which composes it with this price. Nothing is fitted.

Rows. A module that runs per token runs at the rows the step's replayed graph
pads to, `capture_rung` rows of the decode rows' query length, or at the real
rows when built with `padded=False` (the run file's ``padding: none``).
Attention runs over the real rows, a pad row holding no context. The LM head
runs on the rows that sample: each prefill row's last token and every decode
token, none on a step that produces no output.

Terms. One per module kind, `roofline.<kind>`. A term is `declared`, its
note naming each input, when a spec field one of its modules read for the
bound that won, or for its collective, is neither from a datasheet nor
probed, and `analytical` otherwise.

Refused: a step with image tokens, since nothing here prices the vision
encoder that fills them; any step, while the walk found a module it could not
classify; a step whose modules read a spec field the spec does not state.

`StepFallbackBackend` asks a backend first and the roofline only for a step
that backend refuses, so a law with no vector for a step still prices it. The
roofline's term carries what the first backend declined, which shows the
fall-through in the record; the step is answered unless the roofline refuses
it too. Both backends
are step-level: there is no fallback across granularities, so a first backend
of another granularity is refused at construction.
"""

from __future__ import annotations

import dataclasses
from collections import Counter

from atom.compass.backends.base import CostBackend, Granularity
from atom.compass.backends.cost import CostTerm, StepCost, fold_seconds
from atom.compass.backends.provenance import Method, Provenance, Refusal
from atom.compass.backends.resolver import CostRefused
from atom.compass.backends.shape import IMAGES, BatchView, sum_tokens
from atom.compass.backends.work import (
    ELEMENT_BYTES,
    Work,
    allgather,
    allreduce,
    attention,
    bounds,
    elementwise,
    gemm,
)

SOURCE = "step-roofline"
#: The activations' format: the models priced here run them in bf16.
ACT = "bf16"
#: Kinds whose work is one pass over `dims[0]` elements a token.
PASSES = ("embedding", "norm", "rotary", "conv")
UNCLASSIFIED = "unclassified"


@dataclasses.dataclass(frozen=True)
class Module:
    """One module of one rank, as the walk read it off the meta build.

    `kind` picks its work; `dims` are its per-rank shapes in that kind's order:
    `gemm` and `lm_head` (n, k), a pass (elements a token), `act` (output
    elements a token), `attention` (heads, KV heads, head dim),
    `linear_attention` (value heads, key heads, key dim, value dim). `fmt` is
    its weight, KV or state format; `ranks` the group of the collective it
    joins, 1 for none; `fused` that the collective runs in the module's kernel.
    An unclassified module's `name` says what it is.
    """

    name: str
    kind: str
    dims: tuple[int, ...] = ()
    fmt: str = ACT
    ranks: int = 1
    fused: bool = False


def _work(m: Module, view: BatchView, tokens: int, sampled: int) -> Work | None:
    """`m`'s work on a step of `tokens` executed tokens; None if it runs none."""
    d, act = m.dims, ELEMENT_BYTES[ACT]
    if m.kind == "gemm":
        return gemm(tokens, *d, b=m.fmt) + allreduce(m.ranks, tokens * d[0] * act)
    if m.kind == "lm_head":
        if not sampled:
            return None
        gathered = sampled * d[0] * m.ranks * act
        return gemm(sampled, *d, b=m.fmt) + allgather(m.ranks, gathered)
    if m.kind in PASSES:
        # ponytail: one pass over the module's own width; a per-head norm or
        # rotary touches heads times that, left out of this lower bound.
        width = tokens * d[0]
        return elementwise(width, fmt=m.fmt) + allreduce(m.ranks, width * act)
    if m.kind == "act":
        return elementwise(tokens * d[0], reads=2)
    rows = view.requests
    if m.kind == "attention":
        q, kv = [r.query_tokens for r in rows], [r.context_tokens for r in rows]
        return attention(q, kv, d[0], d[2], d[1], kv_fmt=m.fmt)
    # linear_attention: the recurrence reads q, k, v, gate and beta and writes
    # its output once a token, and writes each row's state, reading it where
    # the row has history; two matrix-vector products a token and head.
    vh, kh, dk, dv = d
    real = sum_tokens(rows)
    held = sum(1 for r in rows if r.cached_tokens)
    moved = real * (2 * kh * dk + 2 * vh * dv + 2 * vh) * act
    moved += (len(rows) + held) * vh * dk * dv * ELEMENT_BYTES[m.fmt]
    # ponytail: the state is priced at the bf16 rate, a lower bound on an
    # fp32 accumulation the spec states no rate for.
    return Work({ACT: 4 * real * vh * dk * dv}, moved)


def _reads(work: Work) -> dict[str, list[str]]:
    """The spec fields each bound of `work`'s price reads, as `work.bounds` does."""
    compute = [
        f"device.compute.{f}"
        for fmt in work.flops_by_format
        for f in (f"{fmt}_flops", f"derate.{fmt}")
    ]
    link = "interconnect.intra_node."
    comm = ["domain_devices", "topology", "link_bandwidth_bytes_per_s", "derate"]
    return {
        "floor": ["device.kernel_floor_s"],
        "compute": compute,
        "memory": ["device.memory.bandwidth_curve"],
        "comm": [link + f for f in comm + ["link_latency_s"]] if work.comm else [],
    }


class StepRooflineBackend(CostBackend):
    """A step as the sum of one roofline per module."""

    def __init__(self, machine, modules, *, padded: bool = True) -> None:
        self.machine = machine
        self.padded = padded
        modules = tuple(modules)
        self.unclassified = [m.name for m in modules if m.kind == UNCLASSIFIED]
        # Modules of one shape price alike: count them, price each shape once.
        self.shapes = Counter(dataclasses.replace(m, name="") for m in modules)
        self.count = len(modules)
        self.name = machine.values.get("name", "")
        self._stated_as: dict[str, str] = {}

    @property
    def granularity(self) -> Granularity:
        return Granularity.STEP

    def _stated(self, path: str) -> str:
        """`path` and its origin when it is neither a datasheet's nor probed."""
        if path not in self._stated_as:
            origin = self.machine.origin(path)["origin"]
            spec = f"{path} {origin} in the machine spec {self.name}"
            self._stated_as[path] = "" if origin in ("datasheet", "probed") else spec
        return self._stated_as[path]

    def price(self, batch_view: BatchView) -> StepCost:
        refuse = []
        if batch_view.image_tokens:
            refuse.append(IMAGES.format(batch_view.image_tokens))
        if self.unclassified:
            refuse.append("unclassified modules: " + ", ".join(self.unclassified))
        rows = batch_view.requests
        tokens = sum_tokens(rows)
        decode = batch_view.decode
        if self.padded and batch_view.capture_rung:
            pad = batch_view.capture_rung - len(decode)
            tokens += pad * max(r.query_tokens for r in decode)
        sampled = 0
        if batch_view.produces_output is not False:
            sampled = len(batch_view.prefill) + sum_tokens(decode)
        priced: dict[str, list[float]] = {}
        stated: dict[str, list[str]] = {}
        for m, count in self.shapes.items():
            work = _work(m, batch_view, tokens, sampled)
            if work is None or refuse:
                continue
            reads = _reads(work)
            missing = [
                p for ps in reads.values() for p in ps if p not in self.machine.values
            ]
            if missing:
                refuse.append(
                    f"{m.kind} reads {', '.join(dict.fromkeys(missing))}, "
                    "which the machine spec does not state"
                )
                continue
            b = bounds(work, self.machine)
            kernel = max(("floor", "compute", "memory"), key=b.get)
            won = [kernel, "comm"] if b["comm"] else [kernel]
            seconds = max(b[kernel], b["comm"]) if m.fused else b[kernel] + b["comm"]
            priced.setdefault(m.kind, []).append(seconds * count)
            notes = stated.setdefault(m.kind, [])
            notes += [s for w in won for p in reads[w] if (s := self._stated(p))]
        if refuse:
            reasons = dict.fromkeys(refuse)
            raise CostRefused(batch_view, [Refusal(SOURCE, r) for r in reasons])
        none = "; no successor: the roofline is the last step-level source"
        terms = []
        for kind, seconds in priced.items():
            note = "; ".join(dict.fromkeys(stated[kind]))
            method = Method.DECLARED if note else Method.ANALYTICAL
            detail = f"sum over {kind} modules at {tokens} tokens"
            provenance = Provenance(method, detail, SOURCE, note=note and note + none)
            terms.append(
                CostTerm(f"roofline.{kind}", fold_seconds(seconds), provenance)
            )
        return StepCost(terms)

    def describe(self) -> str:
        return f"step roofline; {self.count} modules"


class StepFallbackBackend(CostBackend):
    """`first`, and the roofline for a step `first` refuses."""

    def __init__(self, first: CostBackend, roofline: StepRooflineBackend) -> None:
        if first.granularity is not roofline.granularity:
            raise ValueError(
                f"a {first.granularity.value}-level backend cannot fall back to "
                f"a {roofline.granularity.value}-level one: fallback runs only "
                "inside one granularity"
            )
        self.first = first
        self.roofline = roofline

    @property
    def granularity(self) -> Granularity:
        return self.first.granularity

    def price(self, batch_view: BatchView) -> StepCost:
        try:
            return self.first.price(batch_view)
        except CostRefused as refused:
            return StepCost(
                [
                    dataclasses.replace(
                        t,
                        provenance=dataclasses.replace(
                            t.provenance, refusals=refused.declined
                        ),
                    )
                    for t in self.roofline.price(batch_view).terms
                ]
            )

    def describe(self) -> str:
        return f"{self.first.describe()}; below it, {self.roofline.describe()}"
