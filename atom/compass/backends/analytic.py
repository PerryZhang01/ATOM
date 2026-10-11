# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Tier 0: a step priced by the roofline, and a backend that falls to it.

`Roofline` prices a whole step, prefill and decode rows alike, from the
machine's derated peaks:

    compute = 2 . P . Sum N_Q / flops
    memory  = (weight bytes + KV bytes per token . (Sum N_KV + Sum N_Q)) / bandwidth
    step    = max(compute, memory, host floor)

P is the parameter count one rank holds. The weights are read once per step,
the KV each row attends over is read and the KV of the tokens computed is
written, one token's KV spanning every paged layer. The host floor is the
declared launches x h of one step. Nothing here was measured on the model, so
the step's one term is `analytical`, named by the bound that won.

`Tiered` asks a backend first and the roofline only for a step that backend
refuses, so a law with no vector for a step still prices it. The roofline's
term carries what the first backend declined, which shows the fall-through
in the record; the step is answered, not refused.
"""

from __future__ import annotations

import dataclasses
import math

from atom.compass.backends.base import CostBackend, Tier
from atom.compass.backends.cost import CostTerm, StepCost
from atom.compass.backends.ladder import CostRefused
from atom.compass.backends.provenance import Provenance, Species
from atom.compass.backends.shape import BatchView, sum_context, sum_tokens

SOURCE = "roofline"


class Roofline(CostBackend):
    """A step as the larger of its compute time, memory time and host floor."""

    def __init__(
        self,
        machine,
        *,
        parameters: int,
        weight_bytes: int,
        kv_token_bytes: int,
        host_floor_s: float,
    ) -> None:
        # ponytail: bf16 peak and GEMM flops only; attention flops, a
        # quantized model's rate and TP collectives are left out until a
        # step that needs them falls to tier 0.
        self.flops = machine.value("device.compute.bf16_flops") * machine.value(
            "device.compute.derate"
        )
        self.bandwidth = machine.value(
            "device.memory.bandwidth_bytes_per_s"
        ) * machine.value("device.memory.derate")
        if min(parameters, weight_bytes, kv_token_bytes) < 1:
            raise ValueError(
                f"a roofline needs parameters, weight bytes and KV bytes per "
                f"token, got {parameters}, {weight_bytes} and {kv_token_bytes}"
            )
        if not math.isfinite(host_floor_s) or host_floor_s < 0.0:
            raise ValueError(
                f"the host floor is {host_floor_s} s; it is a non-negative duration"
            )
        self.parameters = parameters
        self.weight_bytes = weight_bytes
        self.kv_token_bytes = kv_token_bytes
        self.host_floor_s = float(host_floor_s)

    @property
    def tier(self) -> Tier:
        return Tier.ANALYTIC

    def estimate(self, batch_view: BatchView) -> StepCost:
        rows = batch_view.requests
        tokens = sum_tokens(rows)
        moved = self.weight_bytes + self.kv_token_bytes * (sum_context(rows) + tokens)
        bounds = {
            "compute": 2 * self.parameters * tokens / self.flops,
            "memory": moved / self.bandwidth,
            "host": self.host_floor_s,
        }
        bound = max(bounds, key=bounds.get)
        detail = "max of " + ", ".join(f"{k} {v:.6g} s" for k, v in bounds.items())
        return StepCost(
            [
                CostTerm(
                    f"roofline.{bound}",
                    bounds[bound],
                    Provenance(Species.ANALYTICAL, detail, SOURCE),
                )
            ]
        )

    def describe(self) -> str:
        return (
            f"tier-0 roofline; {self.parameters} parameters, "
            f"{self.host_floor_s:g} s host floor"
        )


class Tiered(CostBackend):
    """`first`, and the roofline for a step `first` refuses."""

    def __init__(self, first: CostBackend, roofline: Roofline) -> None:
        self.first = first
        self.roofline = roofline

    @property
    def tier(self) -> Tier:
        return self.first.tier

    def estimate(self, batch_view: BatchView) -> StepCost:
        try:
            return self.first.estimate(batch_view)
        except CostRefused as refused:
            return StepCost(
                [
                    dataclasses.replace(
                        t,
                        provenance=dataclasses.replace(
                            t.provenance, refusals=refused.declined
                        ),
                    )
                    for t in self.roofline.estimate(batch_view).terms
                ]
            )

    def describe(self) -> str:
        return f"{self.first.describe()}; below it, {self.roofline.describe()}"
