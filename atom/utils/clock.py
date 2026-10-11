# SPDX-License-Identifier: MIT
"""The logical-process side of the simulated clock: time requests, sends and release.

One `LPRuntime` per logical process (LP), in the process of its clock owner: the
engine step loop or the frontend event loop. The owner is the only thread that
moves the LP clock and the only one that produces a cross-LP message. It asks the
clock authority for time over an injected connection, ``conn.send((kind, t,
log, t_daemon))`` then ``conn.recv() -> (G, released)``, and every request
carries the sends registered since the previous one, so the authority knows each
message an LP produced before it moves that LP's clock.

A grant to ``+inf`` closes the simulation window (`end_run`) before the process
begins to shut down: the run is finished, and the owner's loop must exit rather
than run its timers at ``+inf``. A clock call after it raises; the wrappers send
shutdown frames unstamped (arrival ``None``), so only a direct `stamp_send` call
returns a ``+inf`` arrival; and `close` raises if the owner leaves before it.

A grant names the messages it releases as ``{channel: [(seq, arrival)]}``.
`_step_through` releases them one at a time in ``(arrival, channel, seq)``
order, setting the LP clock to each arrival. On a ``thread`` channel it then
waits until the receiving thread is back at its wait point, so a handler runs
with the clock at its message's arrival and nothing else in the LP moving. An
``inline`` channel is read by the owner itself at its own receive point, so
there is nothing to wait for. Messages are counted by ``(channel, seq)``, so a
channel need not be FIFO. The ``(channel, seq)`` sets here are only ever tested
for membership, never iterated.

The receive side (`WrappedSocket`, `WrappedPoller`) calls `check_arrival` for
every stamped frame it reads, holds a frame back until `is_released`, and
brackets handing one to ATOM with `handed_over` and `back_at_wait_point`. The
send side stamps each frame with `stamp_send`, directly or through the
`RelayQueue` that stands in for the engine's output queue.
"""

import contextlib
import logging
import math
import os
import pickle
import queue
import threading

import zmq

from atom.compass.clock import ChannelTable, LpId
from atom.compass.clock.channels import ReceiveMode

logger = logging.getLogger("atom")

#: Wall seconds a released message may stay unhandled before one diagnostic is
#: printed. The wait itself goes on: it costs no simulated time.
DIAG_S = 30.0

#: The runtime of the LP this process belongs to, or None on a real run.
_installed: "LPRuntime | None" = None


def install(runtime: "LPRuntime | None") -> None:
    """Make every `now` in this process read `runtime`; None restores the real clock."""
    global _installed
    _installed = runtime


def installed() -> "LPRuntime | None":
    """The runtime installed in this process, or None on a real run."""
    return _installed


def now(real) -> float:
    """The LP clock while a runtime is installed; otherwise `real()`.

    A serving-path read passes the machine clock it read before, so a real run
    reads exactly what it did, and a simulated one never calls it.
    """
    runtime = _installed
    return real() if runtime is None else runtime.read_clock()


#: The station job this thread is serving, as ``job.cur = (station, k)``; the
#: executor running the job sets it and clears it.
job = threading.local()


def current_job_time() -> float | None:
    """The clock inside this thread's station job, or ``None`` outside one."""
    cur = getattr(job, "cur", None)
    return None if cur is None else cur[0].time_in_job(cur[1])


class Straggler(Exception):
    """An unreleased message whose arrival this LP has already released past."""


class _Wakeup(threading.Event):
    """An event `zmq.zmq_poll` can also wait on, through `fd`, beside the sockets."""

    def __init__(self) -> None:
        super().__init__()
        # ponytail: one eventfd per receiving thread, never closed; threads are few
        self.fd = os.eventfd(0, os.EFD_NONBLOCK)

    def set(self) -> None:
        super().set()
        os.eventfd_write(self.fd, 1)

    def clear(self) -> None:
        super().clear()
        with contextlib.suppress(BlockingIOError):
            os.eventfd_read(self.fd)


class LPRuntime:
    def __init__(
        self,
        me: LpId,
        table: ChannelTable,
        conn,
        *,
        owner: threading.Thread | None = None,
        diag_s: float = DIAG_S,
    ) -> None:
        self.me, self.table, self.conn, self.diag_s = me, table, conn, diag_s
        self.owner = owner or threading.current_thread()
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.in_run = False
        self.now = 0.0
        self.send_log: list[tuple[str, int, float]] = []
        self.next_seq = {c.name: 0 for c in table.channels_from(me)}
        into = [c.name for c in table.channels_into(me)]
        self.arrived: dict[str, set[int]] = {ch: set() for ch in into}
        self.released: dict[str, set[int]] = {ch: set() for ch in into}
        self.handled: dict[str, set[int]] = {ch: set() for ch in into}
        # Frames read but not released, (channel, seq) -> arrival: checked at each drain.
        self.unreleased: dict[tuple[str, int], float] = {}
        self.taken_by: dict[threading.Thread, tuple[str, int]] = {}
        self.wakes: dict[threading.Thread, _Wakeup] = {}
        self.endpoints: dict[str, str] = {}  # address -> channel kind it carries
        self.relays: dict[str, RelayQueue] = {}  # channel -> the engine's output relay
        self.calls = 0  # clock calls made, TAR and NER
        self.idle_calls = 0  # `calls` when the step loop last passed its idle point
        # The frontend's event loop, once built: a frame a thread takes is a job on it.
        self.loop = None
        # Clock calls refused off the owner thread, as run-summary reasons
        # ``clock:<call> from <thread>``: a caller such as an output thread may
        # log and drop the exception.
        self.refusals: list[str] = []

    def start_run(self) -> None:
        self.in_run = True

    def end_run(self) -> None:
        self.in_run = False

    # ---- clock owner ----

    def read_clock(self) -> float:
        """The LP clock, or on a thread serving a station job, the time in that job."""
        t = current_job_time()
        return self.now if t is None else t

    def stamp_send(self, ch: str) -> tuple[float, int]:
        """Register one message on `ch` as produced now; returns its ``(arrival, seq)``."""
        self._require_owner("stamp_send")
        with self.lock:
            seq = self.next_seq[ch]
            self.next_seq[ch] = seq + 1
            arrival = self.now + self.table.lookahead(ch)
            self.send_log.append((ch, seq, arrival))
        return arrival, seq

    def advance_to(self, T: float) -> None:
        """Price an event: block until granted `T`, releasing what arrives up to it."""
        self._require_open("advance_to")
        if T < self.now:
            raise ValueError(f"{self.me} cannot advance to {T}, it is at {self.now}")
        self._step_through(*self._ca_call("TAR", T))

    def next_event(self, t: float, t_daemon: float = float("inf")) -> float:
        """Idle until `t`, the daemon deadline `t_daemon` or the earliest arrival,
        whichever the grant is; returns it.

        A daemon deadline is a housekeeping timer that does not keep the run
        alive: it fires only once essential work reaches it.
        """
        self.ask_next_event(t, t_daemon)
        return self.take_grant()

    def ask_next_event(self, t: float, t_daemon: float = float("inf")) -> None:
        """Send `next_event`'s request; `take_grant` waits for its grant. The LP
        clock stands still between the two."""
        self._require_open("next_event")
        self._ca_send("NER", t, t_daemon)

    def take_grant(self) -> float:
        """Wait for the grant of the request `ask_next_event` sent; returns it."""
        G, released = self.conn.recv()
        self._step_through(G, released)
        if G == float("inf"):
            self.end_run()
        return G

    def close(self) -> None:
        """The owner leaves its loop; refused before the ``+inf`` grant."""
        if self.now != float("inf"):
            raise RuntimeError(
                f"{self.me} left its loop at {self.now}, before the +inf grant "
                "finished the run"
            )

    def _require_owner(self, what: str) -> None:
        me = threading.current_thread()
        if me is not self.owner:
            self.refusals.append(f"clock:{what} from {me.name}")
            raise RuntimeError(
                f"{what} from thread {me.name!r}: only the clock "
                f"owner {self.owner.name!r} of {self.me} moves its clock or sends"
            )

    def _require_open(self, what: str) -> None:
        self._require_owner(what)
        if self.now == float("inf"):
            raise RuntimeError(f"{what} from {self.me} after the +inf grant")

    def _ca_call(self, kind: str, t: float, t_daemon: float = float("inf")):
        self._ca_send(kind, t, t_daemon)
        return self.conn.recv()

    def _ca_send(self, kind: str, t: float, t_daemon: float) -> None:
        with self.lock:
            log, self.send_log = self.send_log, []
        self.calls += 1
        self.conn.send((kind, t, log, t_daemon))

    def _step_through(self, G: float, released: dict) -> None:
        order = sorted((a, ch, seq) for ch, msgs in released.items() for seq, a in msgs)
        with self.lock:
            for a, ch, seq in order:
                self.now = max(self.now, a)
                self.released[ch].add(seq)
                self.unreleased.pop((ch, seq), None)
                for w in self.wakes.values():
                    w.set()
                if self.table.recv_mode(ch) is ReceiveMode.INLINE:
                    continue
                reported = False
                while seq not in self.handled[ch]:
                    if not self.cv.wait(self.diag_s) and not reported:
                        logger.warning(
                            "%s: released %s seq %d (arrival %s) is not handled after "
                            "%s wall seconds; still waiting",
                            self.me,
                            ch,
                            seq,
                            a,
                            self.diag_s,
                        )
                        reported = True
            self.now = G
            late = [(a, ch, s) for (ch, s), a in self.unreleased.items() if a < G]
            if late:
                self._straggler(*min(late))

    def _straggler(self, arrival: float, ch: str, seq: int) -> None:
        raise Straggler(
            f"{ch} seq {seq} arrives at {arrival} but {self.me} has released up to "
            f"{self.now} without it (lookahead {self.table.lookahead(ch)})"
        )

    def inline_pending(self) -> bool:
        """Is there a released inline message not yet taken?"""
        with self.lock:
            return any(
                self.released[ch] - self.handled[ch]
                for ch in self.released
                if self.table.recv_mode(ch) is ReceiveMode.INLINE
            )

    # ---- receive side ----

    def check_arrival(self, ch: str, arrival: float, seq: int) -> None:
        """Called once per stamped frame read; frames may arrive in any order."""
        with self.lock:
            if seq in self.arrived[ch]:
                raise RuntimeError(f"{ch} seq {seq} arrived twice")
            self.arrived[ch].add(seq)
            if seq not in self.released[ch]:
                self.unreleased[ch, seq] = arrival
                if arrival < self.now:
                    self._straggler(arrival, ch, seq)

    def my_wakeup(self) -> _Wakeup:
        """This thread's event, set whenever a message is released."""
        with self.lock:
            me = threading.current_thread()
            if me not in self.wakes:
                self.wakes[me] = _Wakeup()
            return self.wakes[me]

    def is_released(self, ch: str, seq: int) -> bool:
        """Caller holds `lock`."""
        return seq in self.released[ch]

    def handed_over(self, ch: str, seq: int) -> None:
        with self.lock:
            self.taken_by[threading.current_thread()] = (ch, seq)
            # The owner waits at this frame's arrival until it is handled.
            arrival = self.now
        if self.loop is not None:
            self.loop.job_begin(arrival)

    def back_at_wait_point(self) -> None:
        if self.loop is not None:
            self.loop.job_end()
        with self.lock:
            taken = self.taken_by.pop(threading.current_thread(), None)
            if taken is not None:
                self.count_done_locked(*taken)

    def count_done_locked(self, ch: str, seq: int) -> None:
        """Caller holds `lock`."""
        self.handled[ch].add(seq)
        self.cv.notify_all()


class UnsentRelayItem(Exception):
    """The output thread took a stamped item off a `RelayQueue` and never sent it."""


class WrappedSocket:
    """One end of a channel's zmq socket, with the ``(channel, arrival, seq)`` header.

    The header is its own frame: first, or after the identity frame on a ROUTER.
    Frames read are held back until the runtime releases them, then handed over
    in ``(arrival, seq)`` order; frames sent outside the run pass straight through.
    """

    def __init__(self, rt: LPRuntime, raw: zmq.Socket, ch: str) -> None:
        self.rt, self.raw, self.ch = rt, raw, ch
        self.relay: RelayQueue | None = None
        self.buf: list[tuple] = []  # (ch, arrival, seq, payload frame) read, not taken

    def send(self, data, **kw) -> None:
        self.send_multipart([data], **kw)

    def send_multipart(self, frames: list, **kw) -> None:
        if not self.rt.in_run:
            stamp = (None, None)
        elif self.relay is not None:
            stamp = self.relay.take_stamp()
        else:
            stamp = self.rt.stamp_send(self.ch)
        hdr = pickle.dumps((self.ch, *stamp))  # arrival None: outside the run
        at = 1 if self.raw.type == zmq.ROUTER else 0
        self.raw.send_multipart([*frames[:at], hdr, *frames[at:]], **kw)

    def recv_multipart(self, *args, **kw) -> list:
        """A frame sent outside the run, ATOM's connect handshake: the frames, header removed."""
        frames = self.raw.recv_multipart(*args, **kw)
        ch, arrival, seq = pickle.loads(
            frames.pop(1 if self.raw.type == zmq.ROUTER else 0)
        )
        if arrival is not None:
            raise RuntimeError(f"{ch} seq {seq} read by recv_multipart, past the clock")
        return frames

    def pull(self) -> None:
        """Read every frame already here, without blocking; check and buffer each."""
        while True:
            try:
                hdr, payload = self.raw.recv_multipart(zmq.NOBLOCK, copy=False)
            except zmq.Again:
                return
            ch, arrival, seq = pickle.loads(hdr.bytes)
            if arrival is not None:
                self.rt.check_arrival(ch, arrival, seq)
            self.buf.append((ch, arrival, seq, payload))
            self.buf.sort(key=lambda f: (-math.inf, 0) if f[1] is None else f[1:3])

    def _first_ready(self) -> int | None:
        with self.rt.lock:
            for k, (ch, arrival, seq, _) in enumerate(self.buf):
                if arrival is None or self.rt.is_released(ch, seq):
                    return k
        return None

    def ready(self) -> bool:
        return self._first_ready() is not None

    def _take(self, copy: bool):
        ch, arrival, seq, payload = self.buf.pop(self._first_ready())
        return ch, arrival, seq, payload.bytes if copy else payload

    def recv(self, copy: bool = True):
        """A receiving thread's wait point, or the clock owner's on an inline channel."""
        if self.rt.table.recv_mode(self.ch) is ReceiveMode.INLINE:
            return self._recv_inline(copy)
        if not self.ready():  # nothing to take: the poller is the wait point
            WrappedPoller(self.rt, [self]).poll()
        ch, arrival, seq, payload = self._take(copy)
        if arrival is not None:
            self.rt.handed_over(ch, seq)
        return payload

    # ---- inline receive: the clock owner takes released frames itself ----

    def settle(self) -> None:
        """Block until every frame released on this channel has physically arrived."""
        while True:
            self.pull()
            with self.rt.lock:
                if self.rt.released[self.ch] <= self.rt.arrived[self.ch]:
                    return
            self.raw.poll()

    def _recv_inline(self, copy: bool):
        while not self.poll():
            pass
        ch, arrival, seq, payload = self._take(copy)
        if arrival is not None:
            with self.rt.lock:
                self.rt.count_done_locked(ch, seq)
        return payload

    def poll(self, timeout_ms: int | None = None) -> bool:
        """A bounded poll is an idle point until ``now + timeout``; zero never idles.

        Outside the run there is no clock to ask: it waits on the socket itself.
        """
        self.settle()
        if not self.ready() and timeout_ms != 0:
            if not self.rt.in_run:
                self.raw.poll(timeout_ms)
            elif timeout_ms is None:
                self.rt.next_event(math.inf)
            else:
                self.rt.next_event(self.rt.now + timeout_ms / 1000.0)
            self.settle()
        return self.ready()


class WrappedPoller:
    """`zmq.Poller` for a receiving thread: the wait point where a frame counts as done.

    Channel sockets count as readable only with a released frame; any other socket
    registered (a shutdown signal) is polled as it is.
    """

    def __init__(self, rt: LPRuntime, socks=()) -> None:
        self.rt, self.socks = rt, list(socks)

    def register(self, sock, flags: int = zmq.POLLIN) -> None:
        self.socks.append(sock)

    def poll(self) -> list[tuple]:
        self.rt.back_at_wait_point()
        wake = self.rt.my_wakeup()
        wrapped = [s for s in self.socks if isinstance(s, WrappedSocket)]
        waits = [(getattr(s, "raw", s), zmq.POLLIN) for s in self.socks]
        while True:
            wake.clear()
            for s in wrapped:
                s.pull()
            ready = [
                s
                for s in self.socks
                if (s.ready() if isinstance(s, WrappedSocket) else s.poll(0))
            ]
            if ready:
                return [(s, zmq.POLLIN) for s in ready]
            zmq.zmq_poll([*waits, (wake.fd, zmq.POLLIN)])


class RelayQueue:
    """The engine's output queue: `put` stamps the send, and the send carries the stamp.

    Only the clock owner puts in the run, and one consumer thread gets. The
    stamp travels with its item, FIFO, so the header is the step's time, not the
    moment the output thread woke. Every item taken in the run must be sent.
    """

    def __init__(self, rt: LPRuntime, wsock: WrappedSocket) -> None:
        self.rt, self.wsock, self.q = rt, wsock, queue.Queue()
        self.pending: tuple | None = None  # stamp of the item taken, not yet sent
        self.consumer: threading.Thread | None = None
        wsock.relay = self

    def put(self, item) -> None:
        stamp = self.rt.stamp_send(self.wsock.ch) if self.rt.in_run else (None, None)
        self.q.put((stamp, item))

    put_nowait = put

    def get(self):
        me = threading.current_thread()
        if self.consumer not in (None, me):
            raise RuntimeError(
                f"{self.wsock.ch}: get from thread {me.name!r}; the relay's one "
                f"consumer is {self.consumer.name!r}"
            )
        self.consumer = me
        if self.pending is not None and self.pending[0] is not None:
            raise UnsentRelayItem(
                f"{self.wsock.ch}: the item stamped (arrival {self.pending[0]}, seq "
                f"{self.pending[1]}) was taken and never sent"
            )
        self.pending, item = self.q.get()
        return item

    def take_stamp(self) -> tuple:
        if self.pending is None:
            raise RuntimeError(f"{self.wsock.ch}: one relay item sent twice")
        stamp, self.pending = self.pending, None
        return stamp


# ---- channel endpoints: named by ATOM's wiring, read by `atom.utils.zmq_shim` ----


def channel_of(rt: LPRuntime, kind: str) -> str:
    """The one channel of `rt`'s LP whose name ends ``:kind``."""
    names = [
        c.name
        for c in rt.table.channels_from(rt.me) + rt.table.channels_into(rt.me)
        if c.name.endswith(":" + kind)
    ]
    if len(names) != 1:
        raise KeyError(
            f"{rt.me} has {len(names)} channels of kind {kind!r} ({names}); a "
            "channel socket names exactly one"
        )
    return names[0]


def name_endpoints(
    dp: int, input_address: str, control_address: str, output_address: str, config=None
) -> None:
    """Record the channel kind each address carries; a real run records nothing.

    A socket this LP binds or connects to a recorded address becomes that
    channel's `WrappedSocket`; the channel is looked up then, so an address no
    socket of this LP opens is never resolved. An engine passes its `config`,
    which holds its pipeline-stage and prefill/decode addresses.
    """
    rt = _installed
    if rt is None:
        return
    tag = f"#dp{dp}"
    kinds = {
        input_address: "request" + tag,
        control_address: "control" + tag,
        output_address: "output" + tag,
    }
    if config is not None:
        pc = config.parallel_config
        kinds |= dict.fromkeys(pc.pp_meta_addrs, "meta")
        kinds |= {
            pc.pp_token_addr: "tokens",
            pc.pp_kv_status_addr: "kv_status",
            config.disagg_p2d_addr: "prefill_done",
            config.disagg_d2p_addr: "block_assignment",
        }
    kinds.pop("", None)  # an address the deployment does not use
    rt.endpoints |= kinds


def relay_queue(dp: int):
    """The engine's output queue: a `RelayQueue` on its output channel, else a `queue.Queue`.

    Its socket is opened later, on the output thread, and takes the relay's
    placeholder when it connects to the output address.
    """
    rt = _installed
    if rt is None:
        return queue.Queue()
    ch = channel_of(rt, f"output#dp{dp}")
    rt.relays[ch] = RelayQueue(rt, WrappedSocket(rt, None, ch))
    return rt.relays[ch]


# ---- the engine step loop: hooks ATOM calls, each a no-op on a real run ----


def charge(reply):
    """A reply whose ``ScheduledBatchOutput.predicted_s`` is set advances the clock by it.

    The reply is what says the call cost time, not the call's name. A real
    forward leaves the field None, and no other reply has it.
    """
    rt = _installed
    if rt is not None and getattr(reply, "predicted_s", None) is not None:
        rt.advance_to(rt.now + reply.predicted_s)
    return reply


def idle(deadline, t_daemon: float = math.inf) -> bool:
    """An idle point of a loop that never blocks; True once the run is finished.

    A pass through the loop that made no clock call ran nothing, so this one
    first waits for the LP's next event: an arrival, or ``deadline()``, the time
    the loop next acts on its own. `t_daemon` is a deadline that does not keep
    the run alive. The loop takes True as its shutdown.
    """
    rt = _installed
    if rt is None:
        return False
    if rt.in_run and rt.calls == rt.idle_calls:
        rt.next_event(deadline(), t_daemon)
    rt.idle_calls = rt.calls
    return rt.now == math.inf


def finish() -> None:
    """Offline teardown: wait for the run to finish before the engines are stopped.

    An engine leaves its loop on the ``+inf`` grant, so a shutdown sent at a
    finite time would take it out before the finish. Outside a run, or once it
    has finished, this asks nothing.
    """
    rt = _installed
    while rt is not None and rt.in_run:
        rt.next_event(math.inf)


def wait_output(q: queue.Queue) -> None:
    """The offline driver's idle point: ask for time until `q` holds an output."""
    rt = _installed
    if rt is None:
        return
    while rt.in_run and q.empty():
        rt.next_event(math.inf)
    if rt.now == math.inf and q.empty():
        raise RuntimeError(f"{rt.me}: the run finished with no output left to get")


def close() -> None:
    """The step loop's process leaves; refused before the ``+inf`` grant."""
    if _installed is not None:
        _installed.close()
