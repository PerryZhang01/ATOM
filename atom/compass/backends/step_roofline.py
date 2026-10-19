# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The step roofline, and the step-level fallback to it.

`StepRooflineBackend` prices a whole step, prefill and decode rows alike, from
the machine's derated peaks:

    compute = 2 . P . Sum N_Q / flops
    memory  = (weight bytes + KV bytes per token . (Sum N_KV + Sum N_Q)) / bandwidth
    step    = max(compute, memory)

P is the parameter count one rank holds. The weights are read once per step,
the KV each row attends over is read and the KV of the tokens computed is
written, one token's KV spanning every paged layer. The step is device busy
time only: host time is the runner's host recurrence, which composes it with
this price. The step's one term is named by the bound that won. It is
`declared`, its note naming each stated input, when an input that bound reads
is stated, and `analytical` otherwise: compute reads `parameter_count` and
`device.compute` (`FIELDS`), memory `device.memory`. A spec field from a
datasheet or probe is not stated, the run file's numbers always are, and the
weights never are: they are read off a meta build, as P is when the run file
states none (`parameters_stated`). A step with image tokens is refused: nothing
here prices the vision encoder that fills them.

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

from atom.compass.backends.base import CostBackend, Granularity
from atom.compass.backends.cost import CostTerm, StepCost
from atom.compass.backends.provenance import Method, Provenance, Refusal
from atom.compass.backends.resolver import CostRefused
from atom.compass.backends.shape import IMAGES, BatchView, sum_context, sum_tokens

SOURCE = "step-roofline"
#: The machine spec fields a roofline reads; `device.<bound>` counts in that bound.
FIELDS = (
    "device.compute.bf16_flops",
    "device.compute.derate.bf16",
    "device.memory.bandwidth_bytes_per_s",
    "device.memory.derate",
)


def _method(origin: str) -> Method:
    """A spec field's method as a roofline parameter."""
    return Method.ANALYTICAL if origin in ("datasheet", "probed") else Method.DECLARED


class StepRooflineBackend(CostBackend):
    """A step as the larger of its compute time and memory time."""

    def __init__(
        self,
        machine,
        *,
        parameters: int,
        weight_bytes: int,
        kv_token_bytes: int,
        parameters_stated: bool = True,
    ) -> None:
        # ponytail: bf16 peak and GEMM flops only; attention flops, a
        # quantized model's rate and TP collectives are left out until a
        # step that needs them falls to the roofline.
        self.flops = machine.value("device.compute.bf16_flops") * machine.value(
            "device.compute.derate.bf16"
        )
        self.bandwidth = machine.value(
            "device.memory.bandwidth_bytes_per_s"
        ) * machine.value("device.memory.derate")
        if min(parameters, weight_bytes, kv_token_bytes) < 1:
            raise ValueError(
                f"a roofline needs parameters, weight bytes and KV bytes per "
                f"token, got {parameters}, {weight_bytes} and {kv_token_bytes}"
            )
        self.parameters = parameters
        self.weight_bytes = weight_bytes
        self.kv_token_bytes = kv_token_bytes
        origins = {path: machine.origin(path)["origin"] for path in FIELDS}
        stated = {"compute": [], "memory": []}
        if parameters_stated:
            stated["compute"].append("parameter_count stated in the run file")
        spec = f"in the machine spec {machine.value('name')}"
        for p, o in origins.items():
            if _method(o) is Method.DECLARED:  # p is device.<bound>.<field>
                stated[p.split(".")[1]].append(f"{p} {o} {spec}")
        none = "; no successor: the roofline is the last step-level source"
        self.note = {b: "; ".join(s) + none for b, s in stated.items() if s}

    @property
    def granularity(self) -> Granularity:
        return Granularity.STEP

    def price(self, batch_view: BatchView) -> StepCost:
        rows = batch_view.requests
        if batch_view.image_tokens:
            reason = IMAGES.format(batch_view.image_tokens)
            raise CostRefused(batch_view, [Refusal(SOURCE, reason)])
        tokens = sum_tokens(rows)
        moved = self.weight_bytes + self.kv_token_bytes * (sum_context(rows) + tokens)
        bounds = {
            "compute": 2 * self.parameters * tokens / self.flops,
            "memory": moved / self.bandwidth,
        }
        bound = max(bounds, key=bounds.get)
        detail = "max of " + ", ".join(f"{k} {v:.6g} s" for k, v in bounds.items())
        method = Method.DECLARED if bound in self.note else Method.ANALYTICAL
        return StepCost(
            [
                CostTerm(
                    f"roofline.{bound}",
                    bounds[bound],
                    Provenance(method, detail, SOURCE, note=self.note.get(bound, "")),
                )
            ]
        )

    def describe(self) -> str:
        return f"step roofline; {self.parameters} parameters"


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
