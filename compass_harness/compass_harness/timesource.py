# SPDX-License-Identifier: MIT
"""Simulated stand-ins for the wall-clock reads aiperf's results depend on.

The package's import hook rebinds ``time`` in the phase lifecycle, the credit
issuer and the agentic replay strategy to ``SimTime``, and ``uuid4`` in the CLI
runner to ``fixed_uuid4``. Nothing here imports aiperf: the hook loads this
module while ``aiperf.timing.phase.runner`` is still importing.
"""

import asyncio
import uuid

#: Added to every simulated ns stamp. Simulated time starts at 0, and aiperf
#: reads a 0 start stamp as missing.
EPOCH_NS = 10**9


def clock():
    from compass_harness.scheduler import ClockPacedLoopScheduler

    if ClockPacedLoopScheduler.clock is None:
        raise RuntimeError(
            "Compass clock not bound: aiperf read the time before the timing "
            "manager built its router."
        )
    return ClockPacedLoopScheduler.clock


def sim_ns(seconds: float) -> int:
    """Simulated seconds as aiperf's ns stamps: the transport's record axis."""
    return round(seconds * 1e9) + EPOCH_NS


class SimTime:
    """``time`` for the modules the hook rebinds: only the calls they make."""

    @staticmethod
    def time_ns() -> int:
        return sim_ns(clock().now())

    perf_counter_ns = time_ns

    @staticmethod
    def monotonic() -> float:
        return clock().now()


class ClockTimer:
    """A one-shot ``callback`` at clock time now + ``delay``, like ``loop.call_later``.

    Its clock wait is withdrawn on ``cancel``, so a dropped timer never asks
    the clock for time. The task waiting is never cancelled.
    """

    def __init__(self, delay: float, callback) -> None:
        self._clock = clock()
        self._at = self._clock.now() + max(0.0, delay)
        self._wait = None
        self.cancelled = False
        asyncio.ensure_future(self._run(callback))

    async def _run(self, callback) -> None:
        while not self.cancelled and self._clock.now() < self._at:
            self._wait = self._clock.wait(self._at)
            await self._wait
        if not self.cancelled:
            callback()

    def cancel(self) -> None:
        self.cancelled = True
        if self._wait is not None:
            self._clock.withdraw(self._wait)


def fixed_uuid4() -> uuid.UUID:
    """``uuid4`` for ``aiperf.cli_runner``: a fixed benchmark id, so the cache-bust
    marker, and with it every prompt's token count, repeats from run to run."""
    return uuid.UUID(int=0)
