# SPDX-License-Identifier: MIT
"""Replay a real run's step times on a simulated one, so both meet the same arrivals.

A real run's record directory holds its step record and `ARRIVALS`, the
worker-clock time each request was sent, which `traffic` writes. The
simulated run sends each request at its offset from the first send, and
`ReplayBackend` ends its rank's step i at the time, on the same offset, that
the real rank's step i returned from its forward, counting steps as `read`
does: the engine loop takes in arrivals and schedules once the forward
returns. A step the record has no line for is refused.
"""

import json
from pathlib import Path

from atom.compass.backends.base import CostBackend, Granularity
from atom.compass.backends.cost import CostTerm, StepCost
from atom.compass.backends.provenance import Method, Provenance, Refusal
from atom.compass.backends.resolver import CostRefused
from atom.compass.parity import ARRIVALS, read

SOURCE = "replay"


class ReplayBackend(CostBackend):
    """One DP rank's real steps, in order: `price` prices step i at its real
    forward's duration, and `charge` makes it end when that forward returned."""

    def __init__(self, directory, dp_rank: int) -> None:
        self.path = Path(directory) / f"dp{dp_rank}.jsonl"
        origin = min(json.loads((Path(directory) / ARRIVALS).read_text())["sent_ns"])
        steps = read(directory)[dp_rank]
        self.ends = [(s["t_exit_ns"] - origin) / 1e9 for s in steps]
        self.seconds = [(s["t_exit_ns"] - s["t_enter_ns"]) / 1e9 for s in steps]
        self.step = 0
        self.now = None  # the engine LP's clock, set where the run installs it

    @property
    def granularity(self) -> Granularity:
        return Granularity.STEP

    def price(self, batch_view) -> StepCost:
        i, self.step = self.step, self.step + 1
        if i >= len(self.seconds):
            why = f"{self.path} has {len(self.seconds)} steps, and this is step {i}"
            raise CostRefused(batch_view, [Refusal(SOURCE, why)])
        detail = f"real step {i}'s forward"
        measured = Provenance(Method.MEASURED, detail, SOURCE)
        return StepCost([CostTerm("forward", self.seconds[i], measured)])

    def charge(self, batch, device_s: float) -> float:
        """Seconds from now to the real return of the step `price` just priced."""
        i = min(self.step, len(self.ends)) - 1
        return max(0.0, self.ends[i] - self.now())

    def describe(self) -> str:
        return f"replay of {self.path}, {len(self.seconds)} steps"
