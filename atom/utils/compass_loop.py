# SPDX-License-Identifier: MIT
"""The frontend LP's tokenizer station: thread-pool jobs on the simulated clock.

Tokenization runs for real, for its result, and is charged a modelled duration
for its time. A `Station` is a multi-server FIFO over that charged time: job k,
submitted at ``t_k``, starts at ``s_k = max(t_k, earliest free server)`` and
completes at ``c_k = s_k + its charged service``. Jobs are placed in submission
order, whatever order their real threads finish in.

`SimExecutor` is the event loop's default executor. From `SimExecutor.start_run`
on, `classify` sorts every job by name: ``do_preprocess`` is a service job on
the station; ``Queue.get`` is a wait job passed straight through, since a
channel event completes it and zero simulated time is right; anything else is
refused, so a job ATOM later adds to the pool is reported, not timed as free.
A service job's result reaches its coroutine at ``c_k``, through ``call_at``.

A thread serving a station job reads the clock as the job's start plus the
service charged so far (`atom.utils.clock.current_job_time`).

A frontend output thread has a width-1 detokenization station of its own:
handling one released frame is one job, from `LPRuntime.handed_over` to its
next `LPRuntime.back_at_wait_point`, charged by the `wrap_decode` calls it
makes. Callbacks it posts with ``call_soon_threadsafe`` run at its completion.
A `wrap_encode` or `wrap_decode` call on the event loop thread, such as the
final decode of a non-streaming completion, advances the LP clock instead.

`CompassEventLoop` is the frontend's event loop in a simulated run: its
``time()`` is the installed `LPRuntime`'s clock, so every timer is on LP time,
and its selector is the frontend's idle point. Idle with nothing to read, it
asks for time with `next_event`: its earliest essential timer as ``t``, its
earliest daemon timer (`DAEMON_TIMERS`) as ``t_daemon``. While a released
request is unread or a station job is open it only waits on its sockets, and
warns once, naming what it waits for, if no socket event ends that wait within
``diag_s`` wall seconds. Its select returns nothing only once ``time()`` has
reached the caller's timeout. The ``+inf`` grant ends the run: from then on
``time()`` runs on from the run's last LP time at wall rate. A uvicorn server
on the loop is asked to exit, as a signal would ask it, and shuts down with
nothing pending; with none, the loop cancels its timers and stops. Over a
clock connection with a socket, it waits for the grant and its own sockets at
once, so a request from outside the run, such as a router's health check, is
served while the grant is out; a timer that request sets earlier than the time
asked for fires at the grant. `HttpChannel` is the inline receive of HTTP
requests.
"""

import asyncio
import collections
import heapq
import logging
import math
import select
import selectors
import sys
import threading
import time
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial

from atom.compass.carriers import relay_stamp, with_relay_stamp
from atom.utils import clock
from atom.utils.clock import DIAG_S, job

logger = logging.getLogger("atom")


class Refused(Exception):
    """An operation the simulation declines; ``args[0]`` is ``(source, name)``."""


class Station:
    """A FIFO with `width` servers, over the service time its jobs are charged."""

    def __init__(self, width: int, diag_s: float = DIAG_S) -> None:
        if width < 1:
            raise ValueError(f"a station needs at least one server, got {width}")
        self.diag_s = diag_s
        self.cv = threading.Condition()
        # One row per job: [submitted, charged, done, start, completion].
        self.jobs: list[list] = []
        self.free = [float("-inf")] * width  # min-heap of server free times
        self.final = 0  # jobs[:final] are placed

    def enqueue(self, t: float) -> int:
        with self.cv:
            self.jobs.append([t, 0.0, False, None, None])
            return len(self.jobs) - 1

    def charge(self, k: int, d: float) -> None:
        with self.cv:
            self.jobs[k][1] += d

    def finish(self, k: int) -> range:
        """Job k is done; returns the jobs this placed, in submission order."""
        with self.cv:
            self.jobs[k][2] = True
            first = self.final
            while self.final < len(self.jobs) and self.jobs[self.final][2]:
                row = self.jobs[self.final]
                row[3] = max(row[0], heapq.heappop(self.free))
                row[4] = row[3] + row[1]
                heapq.heappush(self.free, row[4])
                self.final += 1
            self.cv.notify_all()
            return range(first, self.final)

    def start_of(self, k: int) -> float:
        """Job k's start; blocks until every earlier job is placed."""
        with self.cv:
            if not self.cv.wait_for(lambda: self.final >= k, self.diag_s):
                logger.warning(
                    "station job %d waits for job %d, which is not placed yet after "
                    "%s wall seconds; still waiting",
                    k,
                    self.final,
                    self.diag_s,
                )
                self.cv.wait_for(lambda: self.final >= k)
            s = self.jobs[k][3]
            return max(self.jobs[k][0], self.free[0]) if s is None else s

    def time_in_job(self, k: int) -> float:
        with self.cv:
            return self.start_of(k) + self.jobs[k][1]

    def completion(self, k: int) -> float | None:
        """Job k's completion, or ``None`` until it is placed."""
        with self.cv:
            return self.jobs[k][4]

    def unresolved(self) -> bool:
        with self.cv:
            return self.final < len(self.jobs)


def classify(fn) -> str:
    """``service``, ``wait`` or ``refuse``: the job registry, by qualified name."""
    name = getattr(fn, "__qualname__", "")
    if name.rpartition(".")[2] == "do_preprocess":
        return "service"
    if name == "Queue.get":
        return "wait"
    return "refuse"


class SimExecutor(ThreadPoolExecutor):
    """The loop's default executor; its jobs run in simulated time from `start_run`."""

    def __init__(self, loop, max_workers: int | None = None):
        super().__init__(max_workers)
        self.loop = loop
        self.station: Station | None = None
        self.passthrough = 0  # wait jobs in flight
        self.refusals: list[str] = []  # run-summary reasons, ``executor:<name>``
        self.settle: dict[int, partial | None] = {}  # finished, not yet placed

    def start_run(self) -> None:
        """Freeze the width: the pool's, less the wait jobs resident now."""
        self.station = Station(self._max_workers - self.passthrough)

    def submit(self, fn, /, *args, **kwargs) -> Future:
        kind = "wait" if self.station is None else classify(fn)
        if kind == "refuse":
            name = getattr(fn, "__qualname__", repr(fn))
            self.refusals.append(f"executor:{name}")
            raise Refused(("executor", name))
        if kind == "wait":
            self.passthrough += 1
            cf = super().submit(fn, *args, **kwargs)
            cf.add_done_callback(lambda _: self.loop.call_soon_threadsafe(self._waited))
            return cf
        k = self.station.enqueue(self.loop.time())
        out = Future()
        super().submit(self._serve, k, partial(fn, *args, **kwargs), out)
        return out

    def _waited(self) -> None:
        self.passthrough -= 1

    def _serve(self, k: int, fn, out: Future) -> None:
        settle = None  # a job cancelled before it started is skipped, as the pool does
        if out.set_running_or_notify_cancel():
            job.cur = (self.station, k)
            try:
                settle = partial(out.set_result, fn())
            except BaseException as e:  # noqa: BLE001 - the coroutine gets it
                settle = partial(out.set_exception, e)
            finally:
                job.cur = None
        self.loop.call_soon_threadsafe(self._finish, k, settle)

    def _finish(self, k: int, settle) -> None:
        # On the loop thread, so a placed job's timer is set before the loop's
        # next select can see the station resolved and advance the clock.
        self.settle[k] = settle
        for j in self.station.finish(k):
            done = self.settle.pop(j)
            if done is not None:
                self.loop.call_at(self.station.completion(j), done)


def _charge(d: float) -> None:
    """Charge `d` to this thread's station job; with none open, in the run, the
    caller is the clock owner and advances the LP clock by `d` (`advance_to`
    refuses any other thread)."""
    cur = getattr(job, "cur", None)
    if cur is not None:
        cur[0].charge(cur[1], d)
        return
    rt = clock.installed()
    if rt is not None and rt.in_run:
        rt.advance_to(rt.now + d)


def wrap_encode(encode, entry):
    """`encode`, charging the current job ``fixed + tokens / derated rate``."""
    rate = entry.encode_tokens_per_s * entry.derate

    def charged(*args, **kwargs):
        ids = encode(*args, **kwargs)
        _charge(entry.encode_fixed_s + len(ids) / rate)
        return ids

    return charged


def wrap_decode(decode, entry):
    """`decode`, charging the current job ``fixed + tokens / derated rate``."""
    rate = entry.decode_tokens_per_s * entry.derate

    def charged(ids, *args, **kwargs):
        _charge(entry.decode_fixed_s + len(ids) / rate)
        return decode(ids, *args, **kwargs)

    return charged


#: Timers that do not keep the run alive, by the qualified name of their
#: callback or of a coroutine on the stack that scheduled them: the API
#: server's metrics refresh, uvicorn's server tick and its keep-alive timeout.
DAEMON_TIMERS = frozenset(
    {
        "_metrics_refresh_loop",
        "Server.main_loop",
        "H11Protocol.timeout_keep_alive_handler",
        "HttpToolsProtocol.timeout_keep_alive_handler",
    }
)


def _is_daemon(callback) -> bool:
    if getattr(callback, "__qualname__", None) in DAEMON_TIMERS:
        return True
    f = sys._getframe(2)  # the caller of `call_at`
    while f is not None and f.f_code.co_qualname not in DAEMON_TIMERS:
        f = f.f_back
    return f is not None


class CompassEventLoop(asyncio.SelectorEventLoop):
    """The frontend's event loop on its LP clock; uvicorn's ``loop`` in a simulated run."""

    def __init__(self) -> None:
        self.rt = clock.installed()
        if self.rt is None:
            raise RuntimeError(
                "CompassEventLoop runs on an LP clock; install the frontend's "
                "LPRuntime with atom.utils.clock.install first"
            )
        self.daemon = weakref.WeakSet()  # the daemon timers scheduled
        # Requests read, not handed over: (channel, seq) -> (arrival, wake-up).
        self.held: dict[tuple[str, int], tuple[float, asyncio.Future]] = {}
        # `hand_over` wakes requests in (arrival, seq) order; each takes this
        # lock as it wakes (asyncio locks are fair) and holds it until it is
        # counted handled, so the app gets them in that order.
        self.hand_on = asyncio.Lock()
        # Per output thread: its width-1 station, its open job, and the callbacks
        # that job posted.
        self.detok = threading.local()
        self.ended = None  # at the +inf grant: (last LP time, wall time() then)
        super().__init__(CompassSelector(self))
        self.executor = SimExecutor(self)
        self.set_default_executor(self.executor)
        self.rt.loop = self

    def time(self) -> float:
        if self.ended is None:
            return self.rt.read_clock()
        return self.ended[0] + super().time() - self.ended[1]

    def finish(self, end: float) -> None:
        """The ``+inf`` grant ended the run at LP time `end`."""
        self.ended = (end, super().time())
        self._write_to_self()  # so the select that got the grant reports an fd
        for task in asyncio.all_tasks(self):
            coro = task.get_coro()
            if getattr(coro, "__qualname__", None) == "Server.serve":  # uvicorn's
                coro.cr_frame.f_locals["self"].should_exit = True
                return
        for h in self._scheduled:
            h.cancel()
        self.stop()

    def job_begin(self, arrival: float) -> None:
        """A thread took a released frame: handling it is one job on its station."""
        d = self.detok
        if not hasattr(d, "station"):
            d.station, d.due = Station(1), collections.deque()
        d.k, d.posted = d.station.enqueue(arrival), []
        job.cur = (d.station, d.k)

    def job_end(self) -> None:
        """The thread is back at its wait point: its job's callbacks run at completion."""
        d = self.detok
        posted = getattr(d, "posted", None)
        if posted is None:
            return
        job.cur = d.posted = None
        d.station.finish(d.k)
        d.due.append(posted)
        c = d.station.completion(d.k)
        super().call_soon_threadsafe(self.call_at, c, self._post, d.due)

    def _post(self, due) -> None:
        # Timers that share a deadline fire in any order; each takes the oldest
        # job's callbacks, so a thread's jobs post theirs in completion order.
        for callback, args in due.popleft():
            self.call_soon(callback, *args)

    def call_soon_threadsafe(self, callback, *args, context=None):
        posted = getattr(self.detok, "posted", None)
        if posted is None:
            return super().call_soon_threadsafe(callback, *args, context=context)
        posted.append((callback, args))

    def call_at(self, when, callback, *args, context=None):
        handle = super().call_at(when, callback, *args, context=context)
        if _is_daemon(callback):
            self.daemon.add(handle)
        return handle

    def close(self) -> None:
        super().close()
        self.rt.close()

    def hand_over(self) -> bool:
        """Once every released request has been read, wake the released ones in
        ``(arrival, seq)`` order; returns whether any was woken."""
        rt = self.rt
        with rt.lock:
            if any(not rt.released[ch] <= rt.arrived[ch] for ch, _ in self.held):
                return False
            out = sorted(
                (a, seq, ch)
                for (ch, seq), (a, _) in self.held.items()
                if rt.is_released(ch, seq)
            )
        for _, seq, ch in out:
            self.held.pop((ch, seq))[1].set_result(None)
        if out:
            self._write_to_self()  # so the select that woke them reports an fd
        return bool(out)

    def unread(self) -> tuple[int, str] | None:
        """The released request with the lowest seq not yet read, as
        ``(seq, channel)``, or ``None``."""
        rt = self.rt
        with rt.lock:
            return min(
                (
                    (seq, ch)
                    for ch, got in rt.released.items()
                    for seq in got - rt.arrived[ch]
                ),
                default=None,
            )


class CompassSelector(selectors.DefaultSelector):
    """The loop's one blocking point: a socket wait, or a next-event request."""

    def __init__(self, loop: CompassEventLoop) -> None:
        super().__init__()
        self.loop = loop
        # The wait the loop cannot ask time through: (what, wall time to warn
        # at, or None once warned). It outlives the selects an fd event ends.
        self.stall = None
        self.asked = False  # a next-event request is out, its grant not taken

    def select(self, timeout=None):
        loop, rt = self.loop, self.loop.rt
        if not rt.in_run:  # before the run, or after the +inf grant
            return super().select(timeout)
        if loop.executor.station is None:
            loop.executor.start_run()
        station = loop.executor.station
        # `timeout` is in time() units: it expires on the LP clock.
        deadline = math.inf if timeout is None else loop.time() + max(timeout, 0)
        while True:
            loop.hand_over()
            ready = super().select(0)
            if ready or loop.time() >= deadline:
                return ready
            if rt.inline_pending() or station.unresolved():
                # The LP clock stands still here: only an fd event ends the wait.
                unread = loop.unread()
                what = (
                    f"released {unread[1]} seq {unread[0]} is unread"
                    if unread
                    else (
                        f"station job {station.final} is open"
                        if station.unresolved()
                        else "a released request is read and not handled"
                    )
                )
                wall = time.monotonic()
                if self.stall is None or self.stall[0] != what:
                    self.stall = (what, wall + rt.diag_s)
                warn_at = self.stall[1]
                ready = super().select(None if warn_at is None else warn_at - wall)
                if ready:
                    return ready
                if warn_at is not None:
                    logger.warning(
                        "%s: event loop has had no fd event for %s wall seconds "
                        "while %s; still waiting",
                        rt.me,
                        rt.diag_s,
                        what,
                    )
                    self.stall = (what, None)
                continue
            self.stall = None
            if not self.asked:
                t = t_daemon = math.inf
                for h in loop._scheduled:
                    if h.cancelled():
                        continue
                    if h in loop.daemon:
                        t_daemon = min(t_daemon, h.when())
                    else:
                        t = min(t, h.when())
                if not hasattr(rt.conn, "fileno"):
                    end = rt.now
                    if self._granted(end, rt.next_event(t, t_daemon)):
                        return super().select(0)
                    continue
                rt.ask_next_event(t, t_daemon)
                self.asked = True
            waiting = select.poll()
            for fd in (self, rt.conn):
                waiting.register(fd, select.POLLIN)
            if rt.conn.fileno() in {fd for fd, _ in waiting.poll()}:
                self.asked = False
                end = rt.now
                if self._granted(end, rt.take_grant()):
                    return super().select(0)

    def _granted(self, end: float, g: float) -> bool:
        """Whether grant `g` is ``+inf``, which ends the run at LP time `end`."""
        if g != math.inf:
            return False
        self.loop.finish(end)
        return True


class HttpChannel:
    """ASGI middleware: the frontend's inline receive of HTTP requests.

    `stamp(scope)` is a request's ``(arrival, seq)`` read from its carrier, or
    ``None`` for one sent outside the run, which passes straight through. A
    stamped request waits until it is released and every released request has
    been read; released requests are handed on in ``(arrival, seq)`` order and
    count as handled once their body has been received. The app gets the
    received body messages again from its ``receive``.

    An LP whose requests come in on a ``relay`` channel reads each one's stamp
    from its body's ``kv_transfer_params`` instead. An LP that sends on one
    stamps a send on it for each JSON response whose ``kv_transfer_params`` is an
    object, and writes the stamp there.
    """

    def __init__(self, app, stamp) -> None:
        self.app, self.stamp = app, stamp

    async def __call__(self, scope, receive, send):
        rt = clock.installed()
        relay_in = relay_out = None
        if rt is not None and scope["type"] == "http":
            relay_in = _relay(rt.table.channels_into(rt.me))
            relay_out = _relay(rt.table.channels_from(rt.me))
        if relay_out is not None:
            send = _stamping(send, rt, relay_out)
        if relay_in is not None:
            body, receive = await _read_body(receive)
            got = relay_stamp(body)
        else:
            got = self.stamp(scope) if scope["type"] == "http" else None
        if got is not None:
            loop = asyncio.get_running_loop()
            if not isinstance(loop, CompassEventLoop):
                raise TypeError(
                    f"a stamped request is received on a CompassEventLoop, not on "
                    f"{type(loop).__qualname__}"
                )
            rt = loop.rt
            ch = relay_in or next(
                c.name
                for c in rt.table.channels_into(rt.me)
                if c.name.endswith(":http")
            )
            arrival, seq = got
            rt.check_arrival(ch, arrival, seq)
            wake = loop.create_future()
            loop.held[ch, seq] = (arrival, wake)
            await wake
            async with loop.hand_on:
                # Read the whole body first: a request counts as handled only
                # once nothing about it still waits on the socket.
                body = [await receive()]
                while body[-1].get("more_body"):
                    body.append(await receive())
                with rt.lock:
                    rt.count_done_locked(ch, seq)
            inner = receive

            async def receive():
                return body.pop(0) if body else await inner()

        await self.app(scope, receive, send)


def _relay(channels) -> str | None:
    return next((c.name for c in channels if c.name.endswith(":relay")), None)


async def _read_body(receive):
    """The request's whole body, and a `receive` that hands it on again."""
    chunks, more = [], True
    while more:
        message = await receive()
        chunks.append(message.get("body", b""))
        more = message.get("more_body", False)
    body, replayed = b"".join(chunks), False

    async def replay():
        nonlocal replayed
        if replayed:
            return await receive()
        replayed = True
        return {"type": "http.request", "body": body, "more_body": False}

    return body, replay


def _stamping(send, rt, ch: str):
    """`send`, holding a JSON response back until its body is whole, to stamp it
    for `ch`."""
    start, chunks = None, []

    async def stamping(message):
        nonlocal start
        if message["type"] == "http.response.start" and (
            b"content-type",
            b"application/json",
        ) in message.get("headers", []):
            start = message
            return
        if start is None:
            return await send(message)
        chunks.append(message.get("body", b""))
        if message.get("more_body", False):
            return
        body = with_relay_stamp(b"".join(chunks), lambda: rt.stamp_send(ch))
        headers = [(k, v) for k, v in start["headers"] if k != b"content-length"]
        headers.append((b"content-length", str(len(body)).encode()))
        await send({**start, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    return stamping
