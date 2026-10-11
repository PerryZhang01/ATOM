# SPDX-License-Identifier: MIT
"""aiperf's credit router with the traffic LP on both directions.

Every credit leaves through ``send_credit`` (``CreditIssuer._issue_credit_internal``
is its one call site), so the stamp is taken there; every return reaches the
strategy only through the return callback, so the hold wraps it.
"""

import asyncio

from aiperf.credit.sticky_router import StickyCreditRouter

from compass_harness.scheduler import ClockPacedLoopScheduler
from compass_harness.traffic_lp import TrafficLP
from compass_harness.transport import credit_key


class CompassCreditRouter(StickyCreditRouter):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.traffic = TrafficLP.from_env()
        ClockPacedLoopScheduler.clock = self.traffic
        self.traffic.done.add_done_callback(self._on_finish)

    async def send_credit(self, credit) -> None:
        self.traffic.send(credit_key(credit.phase, credit.phase_index, credit.id))
        await super().send_credit(credit)

    def set_return_callback(self, callback) -> None:
        super().set_return_callback(self.traffic.hold(callback))

    async def wait_for_workers(self, timeout: float) -> None:
        await super().wait_for_workers(timeout)
        await self.traffic.subscribed(len(self._workers))

    def mark_credits_complete(self) -> None:
        super().mark_credits_complete()
        self.traffic.finish()

    def _on_finish(self, done: asyncio.Future) -> None:
        if not done.cancelled() and done.exception() is not None:
            # The traffic LP failed: a request was open at the +inf grant, or one
            # failed mid-stream before it. Stopping the loop fails this service.
            self.exception(f"traffic LP: {done.exception()!r}")
            asyncio.get_running_loop().stop()
