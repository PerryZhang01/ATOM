# SPDX-License-Identifier: MIT
"""Replay a real run's step times on a simulated one, so both meet the same arrivals.

A real run's record directory holds its step record and `ARRIVALS`, the
worker-clock time each request was sent, which `traffic` writes. The
simulated run sends each request at its offset from the first send, and
`ReplayHost` ends its rank's step i at the time, on the same offset, that
the real rank's step i returned from its forward, counting steps as `read`
does: the engine loop takes in arrivals and schedules once the forward
returns. Where the record stamps the engine's next input pull after that
return, the step's `out_s` is the real seconds from the return to the pull,
which the engine LP advances before it takes in arrivals; without one it is
None. A step the record has no line for ends the run: every step after it
would be refused too, so charging it 0 s and going on stops the clock.
"""

import json
from pathlib import Path

from atom.compass.parity import ARRIVALS, read
from atom.compass.runner.overrides import RunnerRefusal

SOURCE = "replay"


class ReplayHost:
    """One DP rank's real steps, in order, as the runner's host in place of
    `HostDeviceRecurrence`: `charge` makes step i end when real step i's
    forward returned, whatever the cost backend priced it at."""

    def __init__(self, directory, dp_rank: int) -> None:
        self.path = Path(directory) / f"dp{dp_rank}.jsonl"
        origin = min(json.loads((Path(directory) / ARRIVALS).read_text())["sent_ns"])
        steps = read(directory)[dp_rank]
        self.ends = [(s["t_exit_ns"] - origin) / 1e9 for s in steps]
        self.outs = [
            (s["t_pull_ns"] - s["t_exit_ns"]) / 1e9 if "t_pull_ns" in s else None
            for s in steps
        ]
        self.out_s = None  # the step `charge` last charged
        self.step = 0
        self.now = None  # the engine LP's clock, set where the run installs it

    def charge(self, batch, device_s: float, draft_s: float) -> float:
        """Seconds from now to the next step's real return; `out_s` becomes
        that step's."""
        i, self.step = self.step, self.step + 1
        if i >= len(self.ends):
            why = f"{self.path} has {len(self.ends)} steps, and this is step {i}"
            raise RunnerRefusal(f"{SOURCE}: {why}")
        self.out_s = self.outs[i]
        return max(0.0, self.ends[i] - self.now())
