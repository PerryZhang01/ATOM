# SPDX-License-Identifier: MIT
"""The traffic LP: aiperf's ``timing_manager`` as one logical process on the Compass clock.

Its clock owner is the event loop thread, which blocks inside each clock call;
worker reports and credit returns queue in ZMQ meanwhile. It is the clock
``ClockPacedLoopScheduler`` paces on, stamps every request the router sends,
and publishes the stamp to the workers.

Before each clock call it holds until aiperf has caught up with the clock:
every stream event the Clock Authority released has been reported by a worker,
every credit whose final event is released has had its return passed to
aiperf (a return that comes before its final event's release waits for it, so
the next turn is scheduled from the final arrival either way), no return is
still inside aiperf's callback, and every task on the loop is waiting on
something. It asks for time only while a request is open, a clock wait (a
pacing timer, a phase deadline, the idle watchdog) is pending or the run is
finishing; between phases aiperf works on the wall clock and the simulation
waits for it.

A request that errors mid-stream never reports the events after its last read.
When released events stay unreported for ``DIAG_S`` wall seconds while an
errored request is still open, the hold fails, naming those requests.
"""

import asyncio
import logging
import math
import os

import zmq
import zmq.asyncio

from atom.compass import clock_transport
from atom.compass import run as compass_run
from atom.compass.clock import LpId
from atom.utils.clock import DIAG_S, LPRuntime
from compass_harness import ADDRESS_ENV
from compass_harness.transport import addresses, credit_key

# The single-engine run's two channels; a run file names its own (`from_env`).
HTTP = "traffic->frontend:http"
STREAM = "frontend->traffic:stream"
TRAFFIC = LpId("traffic")
#: Loop turns the hold waits for aiperf to block before it says so in the log.
SPIN_DIAG = 100_000

logger = logging.getLogger(__name__)


class UnansweredRequests(RuntimeError):
    """Requests that never had their final response."""


class TrafficLP:
    def __init__(
        self, rt: LPRuntime, prefix: str, http: str = HTTP, stream: str = STREAM
    ) -> None:
        self.rt, self.http, self.stream = rt, http, stream
        stamps, reports = addresses(prefix)
        ctx = zmq.asyncio.Context.instance()
        self._pub = ctx.socket(zmq.XPUB)
        self._pub.setsockopt(zmq.XPUB_VERBOSE, 1)
        self._pub.bind(stamps)
        self._pull = ctx.socket(zmq.PULL)
        self._pull.bind(reports)
        self._subscribers = 0
        self.open: dict[tuple, None] = {}  # sent, return not yet passed to aiperf
        self._reports: dict[int, tuple] = {}  # stream seq -> (key, final)
        # key -> its return, for aiperf; None once passed
        self._held: dict[tuple, object] = {}
        self._returns: asyncio.Queue = asyncio.Queue()  # passed in order, as upstream
        self._waiters: list[tuple[float, asyncio.Future]] = []
        self._passing = False  # a return is inside aiperf's callback
        self._finishing = False
        self._changed = asyncio.Event()
        self._tasks = [
            asyncio.ensure_future(self._read_reports()),
            asyncio.ensure_future(self._pass_returns()),
        ]
        self.done = asyncio.ensure_future(self._own())

    @classmethod
    def from_env(cls) -> "TrafficLP":
        """Join the run the deployment's run file describes: its Clock Authority
        at ``clock_endpoint`` and its channel table."""
        run = compass_run.spec()
        if run is None:
            raise RuntimeError(
                f"{compass_run.ENV} is not set: the traffic LP has no run file naming "
                "the Clock Authority and the channel table."
            )
        conn = clock_transport.connect(TRAFFIC, run["clock_endpoint"])
        table = compass_run.channel_table(run)
        # One channel each way: into the frontend (frontend-P behind a PD
        # router) and back from the one that streams (frontend-D).
        (http,), (stream,) = table.channels_from(TRAFFIC), table.channels_into(TRAFFIC)
        return cls(
            LPRuntime(TRAFFIC, table, conn),
            os.environ[ADDRESS_ENV],
            http.name,
            stream.name,
        )

    # ---- the clock ClockPacedLoopScheduler paces on ----

    def now(self) -> float:
        return self.rt.now

    async def advance_to(self, t: float) -> None:
        """Return at the next grant; the scheduler then reads its timers again."""
        await self.wait(t)

    def wait(self, t: float) -> asyncio.Future:
        """A clock wait for ``t``, resolved at the next grant or by ``withdraw``."""
        fut = asyncio.get_running_loop().create_future()
        self._waiters.append((t, fut))
        self._changed.set()
        return fut

    def withdraw(self, fut: asyncio.Future) -> None:
        """Drop a wait, so it asks the clock for nothing, and resolve it now."""
        self._waiters = [w for w in self._waiters if w[1] is not fut]
        if not fut.done():
            fut.set_result(None)
        self._changed.set()

    # ---- the router's side ----

    def send(self, key: tuple) -> None:
        """Stamp one request on the HTTP channel and publish the stamp."""
        arrival, seq = self.rt.stamp_send(self.http)
        self.open[key] = None
        self._pub.send_pyobj((key, arrival, seq))
        self._changed.set()

    def hold(self, callback):
        """aiperf's return callback, behind the hold."""

        async def held(worker_id, message) -> None:
            credit = message.credit
            key = credit_key(credit.phase, credit.phase_index, credit.id)
            if message.error is not None:
                # Passed now, as upstream. A released final event counts as its
                # return; with none, the key stays open and the finish names it.
                self._returns.put_nowait(lambda: callback(worker_id, message))
                self._held[key] = None
            else:
                self._held[key] = lambda: callback(worker_id, message)
            self._changed.set()

        return held

    async def subscribed(self, workers: int) -> None:
        """Wait until `workers` stamp subscriptions have reached the publisher."""
        while self._subscribers < workers:
            if (await self._pub.recv())[:1] == b"\x01":
                self._subscribers += 1

    def finish(self) -> None:
        """aiperf has sent everything: ask for time until the ``+inf`` grant."""
        self._finishing = True
        self._changed.set()

    # ---- the owner ----

    async def _read_reports(self) -> None:
        while True:
            key, seq, arrival, final = await self._pull.recv_pyobj()
            self.rt.check_arrival(self.stream, arrival, seq)
            self._reports[seq] = (key, final)
            self._changed.set()

    async def _pass_returns(self) -> None:
        while True:
            ret = await self._returns.get()
            self._passing = True
            try:
                await ret()
            except Exception:  # upstream loses that one return, not the rest
                logger.exception("traffic: a return callback raised")
            finally:
                self._passing = False
                self._changed.set()

    async def _own(self) -> None:
        while True:
            await self._settle()
            if not (self.open or self._waiters or self._finishing):
                self._changed.clear()
                await self._changed.wait()
                continue
            t = min((t for t, _ in self._waiters), default=math.inf)
            g = self.rt.next_event(t)
            if g == math.inf:
                return self._close()

    async def _settle(self) -> None:
        """Hold until every released event is reported and aiperf has caught up."""
        for _, fut in self._waiters:  # wake the scheduler to read its timers again
            fut.set_result(None)
        self._waiters.clear()
        spins = 0
        while True:
            unreported, unreturned = [], []
            # A handled seq leaves the runtime's released set, so a pass scans
            # only the open events, not every event of the run.
            for seq in sorted(self.rt.released[self.stream]):
                if seq not in self._reports:
                    unreported.append(seq)
                    continue
                key, final = self._reports[seq]
                if final and key not in self._held:
                    unreturned.append(key)
                    continue
                self.rt.released[self.stream].discard(seq)
                if final:
                    self.open.pop(key, None)
                    if (ret := self._held.pop(key)) is not None:
                        self._returns.put_nowait(ret)
            if unreported or unreturned or self._passing:
                self._changed.clear()
                try:
                    await asyncio.wait_for(self._changed.wait(), DIAG_S)
                except TimeoutError:
                    # An errored return leaves its key open with no return to pass.
                    errored = [k for k, ret in self._held.items() if ret is None]
                    if unreported and errored:
                        raise UnansweredRequests(
                            f"requests {errored} failed mid-stream: released stream "
                            f"seqs {unreported} were not reported after {DIAG_S} wall "
                            "seconds"
                        ) from None
                    logger.warning(
                        "traffic at %s: released stream seqs %s not reported, returns "
                        "of %s not received after %s wall seconds; still waiting",
                        self.rt.now,
                        unreported,
                        unreturned,
                        DIAG_S,
                    )
                continue
            ready = _ready_tasks()
            if not ready:
                return
            spins += 1
            if spins == SPIN_DIAG:
                logger.warning("traffic: tasks %s never block; still waiting", ready)
            await asyncio.sleep(0)

    def _close(self) -> None:
        self.rt.close()
        for task in self._tasks:
            task.cancel()
        self._pub.close(linger=0)
        self._pull.close(linger=0)
        if self.open:
            raise UnansweredRequests(
                f"the run finished with requests {list(self.open)} sent but never "
                "given their final response"
            )


def _ready_tasks() -> list[str]:
    """The tasks on this loop, but the current one, not waiting on a pending future."""
    me = asyncio.current_task()
    # ponytail: Task._fut_waiter is private; a loop-level idle hook would not be.
    return [
        task.get_name()
        for task in asyncio.all_tasks()
        if task is not me
        and not task.done()
        and (task._fut_waiter is None or task._fut_waiter.done())
    ]
