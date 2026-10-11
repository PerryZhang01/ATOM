# SPDX-License-Identifier: MIT
"""ATOM's own runner, recording each step: what the frontend names on a real run
with the parity record's variable set.

Kept apart from the package because importing `ModelRunner` needs a driver.
"""

import time

import torch

from atom.compass.parity import StepRecording
from atom.compass.run import host_key
from atom.model_engine.model_runner import ModelRunner


class RecordingModelRunner(StepRecording, ModelRunner):
    """`ModelRunner`, unchanged, with each forward call recorded around it."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Read once at startup, so no forward pays for it.
        self.host_key = host_key(self.config)

    @staticmethod
    def step_clock_ns() -> int:
        """The worker's monotonic clock: a real step's host timestamps."""
        return time.monotonic_ns()

    @staticmethod
    def device_event() -> torch.cuda.Event:
        """A timing event recorded on the current stream: a real step's device stamps."""
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event
