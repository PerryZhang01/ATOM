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
token, none on a step that produces no output. Routed experts run on the
rows the data-parallel group's gather carries (`moe_rows`), and under expert
parallelism on the busiest rank's expected share of them; a graph's pad rows
route as one token.

Two-batch overlap. A step the data-parallel group split (`tbo_tokens`) runs
this rank's tokens as two micro-batches, cut at the first one's token count,
a cut request's second part attending the first as history. Each layer's
modules are priced per micro-batch and laid on one compute and one comm
stream in ATOM's order (`overlap`); the finish time is the `roofline.tbo`
term. Modules outside the layers run once over the whole step.

Terms. One per module kind, `roofline.<kind>`. A term is `declared`, its
note naming each input, when a spec field one of its modules read for the
bound that won, or for its collective, is neither from a datasheet nor
probed, and `analytical` otherwise.

Refused: a step with image tokens, since nothing here prices the vision
encoder that fills them; any step, while the walk found a module it could not
classify; a step whose modules read a spec field the spec does not state; a
data-parallel MoE step with no `moe_rows`; a split step whose counts name no
single pair for this rank, or whose MoE runs MoRI's split send and receive,
whose overlap no stream carries.

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
from math import comb

import numpy as np

from atom.compass.backends.base import CostBackend, Granularity
from atom.compass.backends.cost import CostTerm, StepCost, fold_seconds
from atom.compass.backends.provenance import Method, Provenance, Refusal
from atom.compass.backends.resolver import CostRefused
from atom.compass.backends.shape import IMAGES, BatchView, RequestShape, sum_tokens
from atom.compass.backends.work import (
    ELEMENT_BYTES,
    Work,
    activated_experts,
    all_to_all,
    allgather,
    allreduce,
    attention,
    bounds,
    elementwise,
    gemm,
    grouped_gemm,
    max_rank_load,
    price,
    reduce_scatter,
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
    `linear_attention` (value heads, key heads, key dim, value dim), `mhc`
    (residual copies, width), `sparse_attention` (heads, head dim, window,
    compression ratio, indexer top k, heads and head dim), `moe` as
    `run.modules` reads it. `fmt` is its weight, KV or state format, `act` the
    format its matrix products read activations in; `ranks` the group of the
    collective it joins, 1 for none; `fused` that the collective runs in the
    module's kernel; `layer` the decoder layer it sits in, -1 outside them.
    An unclassified module's `name` says what it is.
    """

    name: str
    kind: str
    dims: tuple[int, ...] = ()
    fmt: str = ACT
    ranks: int = 1
    fused: bool = False
    act: str = ACT
    layer: int = -1


def _work(m: Module, view: BatchView, tokens: int, sampled: int) -> Work | None:
    """`m`'s work on a step of `tokens` executed tokens; None if it runs none."""
    d, act = m.dims, ELEMENT_BYTES[ACT]
    if m.kind == "gemm":
        return gemm(tokens, *d, m.act, m.fmt) + allreduce(m.ranks, tokens * d[0] * act)
    if m.kind == "moe":
        return sum(_moe(m, view, tokens), Work())
    if m.kind == "mhc":
        # Fold the sub-layer's output into the widened residual and reduce it:
        # read and write both, and the mixing weights' fp32 projection.
        copies, width = d
        mix = (2 + copies) * copies
        moved = 2 * tokens * (copies + 1) * width * act
        moved += mix * copies * width * ELEMENT_BYTES["fp32"]
        return Work({ACT: 2 * tokens * mix * copies * width}, moved)
    if m.kind == "sparse_attention":
        return _sparse_attention(m, view.requests)
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


def _moe(m: Module, view: BatchView, tokens: int) -> tuple[Work, Work, Work]:
    """The exchange before the routed experts, the experts, and the exchange
    after, routing uniform and independent.

    The experts run on the gathered rows, or under expert parallelism on the
    busiest rank's expected share, reading each expert a row reaches once.
    On one data-parallel rank the output is all-reduced. Over a group, the
    hidden states and router logits are gathered and the output scattered
    back; under expert parallelism MoRI sends each of this rank's tokens once
    to every rank holding one of its experts, and back.
    """
    experts, local, k, inter, hidden, ranks, dp, _ = m.dims
    rows, act = view.moe_rows if dp > 1 else tokens, ELEMENT_BYTES[ACT]
    real = view.moe_tokens if dp > 1 else sum_tokens(view.requests)
    # Pad rows all carry token 0 at position 0, so they route as one token.
    distinct = rows if real is None else min(rows, real + 1)
    routed, groups = rows * k, activated_experts(experts, k, distinct)
    if local < experts:
        routed, groups = max_rank_load(rows, k, experts, ranks), groups / ranks
    work = grouped_gemm(routed, 2 * inter, hidden, groups, m.act, m.fmt)
    work += grouped_gemm(routed, hidden, inter, groups, m.act, m.fmt)
    if dp == 1:
        return Work(), work, allreduce(ranks, tokens * hidden * act)
    if local == experts:
        gather = allgather(dp, rows * (hidden + experts) * act)
        return gather, work, reduce_scatter(dp, rows * hidden * act)
    sent = tokens * hidden * ranks * (1 - comb(experts - local, k) / comb(experts, k))
    dispatch = all_to_all(ranks, sent * ELEMENT_BYTES[m.act])
    return dispatch, work, all_to_all(ranks, sent * act)


def _prefix(n: np.ndarray, ratio: int = 1, cap: int | None = None) -> np.ndarray:
    """`sum(min(x // ratio, cap) for x in 1..n)`, for each `n`."""
    if cap is not None:
        edge = cap * ratio - 1
        return _prefix(np.minimum(n, edge), ratio) + cap * np.maximum(0, n - edge)
    whole = n // ratio
    return ratio * whole * (whole - 1) // 2 + whole * (n - whole * ratio + 1)


def _sparse_attention(m: Module, rows) -> Work:
    """V4's attention: a query at position `p` reads the last `window` tokens
    and the `p // ratio` compressed entries before it, or the indexer's top
    `k` of them, which the indexer scores against every compressed entry. The
    KV each request reads is read once."""
    heads, dim, window, ratio, top, index_heads, index_dim = m.dims
    seen = np.array([r.context_tokens for r in rows])
    held = seen - np.array([r.query_tokens for r in rows])

    def span(**k) -> int:
        return int((_prefix(seen, **k) - _prefix(held, **k)).sum())

    keys, read = span(cap=window), np.minimum(seen, window).sum()
    work = Work()
    if ratio:
        entries = seen // ratio
        if index_heads:
            scores = 2 * index_heads * index_dim * span(ratio=ratio)
            index = int(entries.sum()) * index_dim * ELEMENT_BYTES[m.fmt]
            work = Work({m.fmt: scores}, index)
            entries = np.minimum(entries, top)
        keys += span(ratio=ratio, cap=top if index_heads else None)
        read += entries.sum()
    q = sum_tokens(rows) * heads * dim * ELEMENT_BYTES[ACT]
    flops = {ACT: 4 * heads * dim * keys}
    return work + Work(flops, 2 * q + int(read) * dim * ELEMENT_BYTES[m.fmt])


def overlap(layers) -> float:
    """The finish time of two micro-batches' layers on one compute and one
    comm stream, in ATOM's issue order.

    Each layer is four pairs of seconds, one per micro-batch: the compute
    before the MoE exchange, the exchange in, the routed experts, the exchange
    out. Within a layer every segment is issued for both micro-batches before
    the next, compute and exchange alternating streams; a segment starts when
    its stream is free and its micro-batch's previous segment is done.
    """
    free = {"compute": 0.0, "comm": 0.0}
    done = [0.0, 0.0]
    for layer in layers:
        for segment, stream in zip(layer, ("compute", "comm", "compute", "comm")):
            for k in 0, 1:
                free[stream] = done[k] = max(free[stream], done[k]) + segment[k]
    return max(free.values())


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
        self.shapes = Counter(
            dataclasses.replace(m, name="", layer=-1)
            for m in modules
            if m.kind != UNCLASSIFIED
        )
        self.outside = Counter(
            dataclasses.replace(m, name="")
            for m in modules
            if m.kind != UNCLASSIFIED and m.layer < 0
        )
        self.layered = [m for m in modules if m.layer >= 0]
        self.count = len(modules)
        self.name = machine.values.get("name", "")
        self.domain = machine.values.get("interconnect.intra_node.domain_devices")
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

    def _unit(self, m: Module, view: BatchView, tokens: int, sampled: int):
        """`m`'s seconds as (exchange before, the rest, exchange after) with
        the notes of the inputs that decided them, a refusal's text, or None
        if `m` runs nothing on this step."""
        if m.kind == "moe" and m.dims[6] > 1 and view.moe_rows is None:
            return f"a MoE over {m.dims[6]} data-parallel ranks, and no moe_rows"
        if m.kind == "moe":
            parts = _moe(m, view, tokens)
        else:
            parts = (Work(), _work(m, view, tokens, sampled), Work())
            if parts[1] is None:
                return None
        work = sum(parts, Work())
        reads = _reads(work)
        missing = [
            p for ps in reads.values() for p in ps if p not in self.machine.values
        ]
        if missing:
            return (
                f"{m.kind} reads {', '.join(dict.fromkeys(missing))}, "
                "which the machine spec does not state"
            )
        if work.comm and m.ranks > self.domain:
            return (
                f"a {m.kind} collective over {m.ranks} ranks leaves the "
                f"scale-up domain of {self.domain} devices"
            )
        b = bounds(work, self.machine)
        kernel = max(("floor", "compute", "memory"), key=b.get)
        won = [kernel, "comm"] if b["comm"] else [kernel]
        notes = [s for w in won for p in reads[w] if (s := self._stated(p))]
        if m.fused:
            return (0.0, price(work, self.machine), 0.0), notes
        comm = [bounds(w, self.machine)["comm"] if w.comm else 0.0 for w in parts]
        return (comm[0], b[kernel] + comm[1], comm[2]), notes

    def _overlapped(self, view: BatchView, refuse, priced, stated) -> str:
        """Price the layers of a split step as two micro-batches on two
        streams into `priced["tbo"]`, or add why not to `refuse`."""
        if any(m.kind == "moe" and m.dims[7] for m in self.layered):
            refuse.append(
                "a two-batch-overlap step on MoRI's split send and receive "
                "(--all2all-backend low-latency): its transfer in flight lies "
                "on no stream"
            )
            return ""
        own = sum_tokens(view.requests)
        pairs = list(dict.fromkeys(p for p in zip(*view.tbo_tokens) if sum(p) == own))
        if len(pairs) != 1:
            refuse.append(
                f"the two-batch-overlap split {view.tbo_tokens} names no single "
                f"pair of micro-batches for this rank's {own} tokens"
            )
            return ""
        # ponytail: the micro-batches run their real rows; a split decode
        # graph's padding is left out of this lower bound.
        left, halves = pairs[0][0], ([], [])
        for r in view.requests:
            first = min(left, r.query_tokens)
            left -= first
            if first:
                halves[0].append(RequestShape(first, r.cached_tokens + first, r.decode))
            if r.query_tokens > first:
                rest = r.query_tokens - first
                halves[1].append(RequestShape(rest, r.context_tokens, r.decode))
        views = [
            BatchView(tuple(h), moe_rows=sum(t))
            for h, t in zip(halves, view.tbo_tokens)
        ]
        layers: dict[int, list[list[float]]] = {}
        units: dict = {}
        notes = stated.setdefault("tbo", [])
        for m in self.layered:
            shape = dataclasses.replace(m, name="", layer=-1)
            segments = layers.setdefault(m.layer, [[0.0, 0.0] for _ in range(4)])
            for k, half in enumerate(views):
                if (shape, k) not in units:
                    unit = self._unit(shape, half, sum_tokens(half.requests), 0)
                    units[shape, k] = unit
                    if isinstance(unit, str):
                        refuse.append(unit)
                    elif unit:
                        notes += unit[1]
                unit = units[shape, k]
                if not unit or isinstance(unit, str):
                    continue
                if m.kind == "moe":
                    for s, seconds in zip(segments[1:], unit[0]):
                        s[k] += seconds
                else:
                    segments[0][k] += sum(unit[0])
        priced["tbo"] = [overlap(layers[n] for n in sorted(layers))]
        return " and ".join(str(sum_tokens(v.requests)) for v in views)

    def price(self, batch_view: BatchView) -> StepCost:
        refuse = []
        if batch_view.image_tokens:
            refuse.append(IMAGES.format(batch_view.image_tokens))
        if self.unclassified:
            refuse.append("unclassified modules: " + ", ".join(self.unclassified))
        if refuse:
            raise CostRefused(batch_view, [Refusal(SOURCE, r) for r in refuse])
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
        split = batch_view.tbo_tokens is not None
        for m, count in (self.outside if split else self.shapes).items():
            unit = self._unit(m, batch_view, tokens, sampled)
            if isinstance(unit, str):
                refuse.append(unit)
            elif unit:
                priced.setdefault(m.kind, []).append(sum(unit[0]) * count)
                stated.setdefault(m.kind, []).extend(unit[1])
        if split:
            split = self._overlapped(batch_view, refuse, priced, stated)
        if refuse:
            reasons = dict.fromkeys(refuse)
            raise CostRefused(batch_view, [Refusal(SOURCE, r) for r in reasons])
        none = "; no successor: the roofline is the last step-level source"
        terms = []
        for kind, seconds in priced.items():
            note = "; ".join(dict.fromkeys(stated[kind]))
            method = Method.DECLARED if note else Method.ANALYTICAL
            detail = f"sum over {kind} modules at {tokens} tokens"
            if kind == "tbo":
                detail = f"the layers' two micro-batches of {split} tokens overlapped"
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
