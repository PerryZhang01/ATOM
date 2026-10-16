# SPDX-License-Identifier: MIT
"""The runner the bench starts on every worker: ATOM's recording runner with the
bench's window.

Kept apart from the package because importing `ModelRunner` needs a driver.
"""

import torch
from aiter.dist.parallel_state import get_world_group

from atom.compass.bench.window import BenchWindow
from atom.compass.parity.runner import RecordingModelRunner

#: Spin cycles timed once at startup, to turn a preload's nanoseconds into cycles.
CALIBRATION_CYCLES = 10_000_000


class BenchModelRunner(BenchWindow, RecordingModelRunner):
    """`RecordingModelRunner`, measuring a batch when asked for `bench_step`."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The first launch also loads the kernel, which takes longer than it spins.
        torch.cuda._sleep(1)
        start = self.device_event()
        torch.cuda._sleep(CALIBRATION_CYCLES)
        end = self.device_event()
        end.synchronize()
        self.cycles_per_ns = CALIBRATION_CYCLES / (start.elapsed_time(end) * 1e6)

    def preload(self, ns: int) -> None:
        """Queue about `ns` of device work on the current stream: a spin kernel."""
        torch.cuda._sleep(round(ns * self.cycles_per_ns))

    @staticmethod
    def bench_group():
        """The workers' CPU group; None for one worker.

        The bench refuses data and pipeline parallelism, so the world is this
        engine's workers.
        """
        group = get_world_group()
        return group.cpu_group if group.world_size > 1 else None
