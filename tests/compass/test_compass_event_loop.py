# SPDX-License-Identifier: MIT
"""`CompassEventLoop` against the co-hosted clock authority.

Three LPs as in a one-engine run. The frontend is the loop under test; the
engine waits in NER(+inf), and so does the traffic LP unless a test drives it.
The daemon timers are the real ones: the API server's metrics refresh loop,
and uvicorn's server tick and keep-alive timeout.
"""

import asyncio
import copy
import http.client
import importlib.util
import json
import logging
import math
import queue
import socket
import threading
import time
import uuid
import warnings
from types import SimpleNamespace

import fastapi
import pytest
import uvicorn

from atom.compass import carriers, clock_transport
from atom.compass.clock import NER, ClockAuthority, LpId, single_engine_table
from atom.entrypoints.openai import api_server
from atom.utils import clock
from atom.utils.clock import LPRuntime
from atom.utils.compass_loop import (
    CompassEventLoop,
    CompassSelector,
    HttpChannel,
    wrap_encode,
)

TABLE = single_engine_table(admission_path="serving", ipc_s=0.001, stream_s=0.002)
TRAFFIC, FRONTEND, ENGINE = LpId("traffic"), LpId("frontend"), LpId("engine")
HTTP = "traffic->frontend:http"
INF = math.inf


class _Watched:
    """The frontend's connection: each request, each grant, and each NER sent
    while a released request was unread."""

    def __init__(self, conn) -> None:
        self.conn, self.sent, self.grants, self.unread = conn, [], [], []
        self.rt = None

    def send(self, msg) -> None:
        if self.rt.inline_pending():
            self.unread.append(msg)
        self.sent.append(msg)
        if len(self.sent) > 1000:
            raise RuntimeError("1000 time requests: a timer keeps the run alive")
        self.conn.send(msg)

    def recv(self):
        g, released = self.conn.recv()
        self.grants.append(g)
        return g, released


@pytest.fixture
def run(monkeypatch):
    endpoint = f"inproc:test-compass-loop-{uuid.uuid4().hex}"
    server = clock_transport.serve(ClockAuthority(TABLE), endpoint)
    clock_transport.connect(ENGINE, endpoint).send((NER, INF, [], INF))
    conn = _Watched(clock_transport.connect(FRONTEND, endpoint))
    conn.rt = rt = LPRuntime(FRONTEND, TABLE, conn)
    rt.start_run()
    clock.install(rt)
    # Every loop the test builds, uvicorn's included, is stopped after 20 wall
    # seconds, and the test fails: a loop that never stops cannot hang it.
    loops, fired, init = [], [], CompassEventLoop.__init__

    def tracked(self):
        init(self)
        loops.append(self)

    def stop_all():
        fired.append(True)
        for loop in loops:
            if not loop.is_closed():
                loop.call_soon_threadsafe(loop.stop)

    monkeypatch.setattr(CompassEventLoop, "__init__", tracked)
    guard = threading.Timer(20, stop_all)
    guard.start()
    try:
        yield SimpleNamespace(rt=rt, conn=conn, endpoint=endpoint)
    finally:
        guard.cancel()
        clock.install(None)
        server.close()
    assert not fired, "a loop ran 20 wall seconds without stopping"


def _idle_traffic(run) -> None:
    clock_transport.connect(TRAFFIC, run.endpoint).send((NER, INF, [], INF))


def _until(cond) -> None:
    deadline = time.monotonic() + 10
    while not cond():
        assert time.monotonic() < deadline, "not reached in 10 wall seconds"
        time.sleep(0.001)


def _traffic(run, stamps: queue.Queue, before_ner, n=1) -> threading.Thread:
    """The traffic LP: stamps `n` requests at 0, then `before_ner()`, then idles."""

    def body():
        rt = LPRuntime(TRAFFIC, TABLE, clock_transport.connect(TRAFFIC, run.endpoint))
        rt.start_run()
        for _ in range(n):
            stamps.put(rt.stamp_send(HTTP))
        before_ner()
        rt.next_event(INF)
        rt.close()

    thread = threading.Thread(target=body, name="traffic", daemon=True)
    thread.start()
    return thread


def _stamp(scope):
    """Stands in for the request carrier: ``x-test-stamp: <arrival> <seq>``."""
    for name, value in scope["headers"]:
        if name == b"x-test-stamp":
            arrival, seq = value.split()
            return float(arrival), int(seq)
    return None


async def _no_body():
    """An ASGI ``receive`` for a request with an empty body."""
    return {"type": "http.request", "body": b"", "more_body": False}


def test_a_60_second_sleep_ends_at_lp_60_with_no_wall_wait(run):
    _idle_traffic(run)
    loop = CompassEventLoop()
    wall = time.monotonic()
    loop.run_until_complete(asyncio.sleep(60))
    wall, lp = time.monotonic() - wall, loop.time()
    loop.run_forever()  # nothing is left: the finish grants +inf and the loop stops
    loop.close()
    print(f"\n  asyncio.sleep(60): {lp} LP seconds in {wall:.4f} wall seconds")
    assert lp == 60.0 and wall < 5.0
    assert run.conn.sent == [(NER, 60.0, [], INF), (NER, INF, [], INF)]
    assert run.rt.now == INF


def test_the_metrics_refresh_is_a_daemon_and_the_finish_stops_the_loop(run):
    _idle_traffic(run)
    refreshed, late = [], []

    async def refresh():
        loop = asyncio.get_running_loop()
        refreshed.append(loop.time())
        # A plain callback, a daemon because the periodic job schedules it.
        loop.call_later(4, lambda: late.append(loop.time()))

    async def serve():
        asyncio.create_task(api_server._periodic(5, refresh))
        await asyncio.sleep(12)  # the last essential timer
        await asyncio.Event().wait()  # serves until stopped, as uvicorn does

    # asyncio.run, as uvicorn calls it: a stop ends the main task early.
    with pytest.raises(RuntimeError, match="Event loop stopped before Future"):
        asyncio.run(serve(), loop_factory=CompassEventLoop)
    # The timers due at 14 and 15 never run, not even at +inf.
    assert refreshed == [5.0, 10.0] and late == [9.0]
    assert run.conn.sent == [
        (NER, 12.0, [], 5.0),
        (NER, 12.0, [], 9.0),
        (NER, 12.0, [], 10.0),
        (NER, 12.0, [], 14.0),
        (NER, INF, [], 14.0),
    ]
    assert run.conn.grants == [5.0, 9.0, 10.0, 12.0, INF]


def test_no_time_is_asked_while_a_station_job_is_open(run):
    _idle_traffic(run)
    loop = CompassEventLoop()

    async def go():
        later = asyncio.ensure_future(asyncio.sleep(1))  # a timer to jump to

        def do_preprocess():
            time.sleep(0.2)  # open across the loop's next select

        await loop.run_in_executor(None, do_preprocess)
        done = loop.time()
        await later
        return done

    assert loop.run_until_complete(go()) == 0.0
    assert len(loop.executor.station.jobs) == 1  # the default executor's station
    loop.run_forever()
    loop.close()


def test_closing_the_loop_before_the_finish_is_refused(run):
    _idle_traffic(run)
    loop = CompassEventLoop()
    loop.run_until_complete(asyncio.sleep(1))
    with pytest.raises(RuntimeError, match=r"before the \+inf grant"):
        loop.close()


def test_a_request_released_before_it_is_read_is_handed_over_at_its_arrival(run):
    stamps, handled, got = queue.Queue(), [], {}

    async def app(scope, receive, send):
        handled.append(asyncio.get_running_loop().time())
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    server = uvicorn.Server(
        uvicorn.Config(
            HttpChannel(app, _stamp),
            host="127.0.0.1",
            port=0,
            loop="atom.utils.compass_loop:CompassEventLoop",
            lifespan="off",
            log_level="warning",
        )
    )
    traffic = _traffic(run, stamps, before_ner=lambda: None)
    arrival, seq = stamps.get(timeout=10)

    def released():
        with run.rt.lock:
            return run.rt.is_released(HTTP, seq)

    def client():
        _until(lambda: server.started and released())
        port = server.servers[0].sockets[0].getsockname()[1]
        got["conn"] = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        got["conn"].request("GET", "/", headers={"x-test-stamp": f"{arrival!r} {seq}"})
        # The connection stays open, so its keep-alive timer stays set.
        got["body"] = got["conn"].getresponse().read()

    thread = threading.Thread(target=client, name="client", daemon=True)
    thread.start()
    try:
        server.run()  # the finish ends it with nothing pending
    finally:
        thread.join(10)
        traffic.join(10)
        if "conn" in got:
            got["conn"].close()
    assert got["body"] == b"ok" and handled == [arrival]
    assert run.conn.unread == []
    # The server tick and the keep-alive timeout never move the clock.
    assert max(g for g in run.conn.grants if g < INF) == arrival
    assert run.conn.grants[-1] == INF


def test_the_finish_shuts_uvicorn_down_with_no_error_and_no_join_warning(
    run, caplog, monkeypatch
):
    _idle_traffic(run)
    # An earlier server may have left uvicorn's log config: no propagation, WARNING.
    monkeypatch.setattr(logging.getLogger("uvicorn"), "propagate", True)
    server = uvicorn.Server(
        uvicorn.Config(
            fastapi.FastAPI(),
            host="127.0.0.1",
            port=0,
            loop="atom.utils.compass_loop:CompassEventLoop",
            lifespan="on",
            log_config=None,
        )
    )
    with (
        caplog.at_level(logging.INFO, "uvicorn.error"),
        warnings.catch_warnings(record=True) as warned,
    ):
        warnings.simplefilter("always")
        try:
            server.run()
            stopped = None
        except RuntimeError as e:  # the loop stopped with uvicorn's task pending
            stopped = str(e)
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    joins = [str(w.message) for w in warned if w.category is RuntimeWarning]
    assert (stopped, errors, joins) == (None, [], [])
    assert run.rt.now == INF and "Application shutdown complete." in caplog.messages


def test_a_request_read_before_its_release_is_held_until_it(run):
    stamps, handled = queue.Queue(), []
    traffic = _traffic(
        run, stamps, before_ner=lambda: _until(lambda: run.rt.arrived[HTTP])
    )
    arrival, seq = stamps.get(timeout=10)

    async def app(scope, receive, send):
        handled.append(asyncio.get_running_loop().time())

    scope = {
        "type": "http",
        "headers": [(b"x-test-stamp", f"{arrival!r} {seq}".encode())],
    }
    loop = CompassEventLoop()
    try:
        loop.run_until_complete(HttpChannel(app, _stamp)(scope, _no_body, None))
        loop.run_forever()
    finally:
        traffic.join(10)
    loop.close()
    assert handled == [arrival]
    assert run.conn.grants == [arrival, INF] and run.conn.unread == []


def test_a_body_that_arrives_after_the_next_timer_is_handed_on_at_its_release(run):
    stamps, handled = queue.Queue(), []
    traffic = _traffic(
        run, stamps, before_ner=lambda: _until(lambda: run.rt.arrived[HTTP])
    )
    arrival, seq = stamps.get(timeout=10)
    loop = CompassEventLoop()
    rest = loop.create_future()  # the body's last part, sent late
    parts = [
        {"type": "http.request", "body": b"a", "more_body": True},
        {"type": "http.request", "body": b"b", "more_body": False},
    ]

    async def receive():
        if len(parts) == 1:
            await rest
        return parts.pop(0) if parts else {"type": "http.disconnect"}

    async def app(scope, receive, send):
        got = [await receive()]
        while got[-1]["more_body"]:
            got.append(await receive())
        handled.append((loop.time(), b"".join(m["body"] for m in got)))

    def deliver():
        # The rest of the body comes once the clock passes the arrival, or
        # after 0.3 wall seconds with the clock standing at it.
        def released():
            with run.rt.lock:
                return run.rt.is_released(HTTP, seq)

        _until(released)
        end = time.monotonic() + 0.3
        while run.rt.now <= arrival and time.monotonic() < end:
            time.sleep(0.001)
        loop.call_soon_threadsafe(rest.set_result, None)

    async def go():
        async def tick():  # the frontend's next timers, as its 0.1 s tick
            while not handled:
                await asyncio.sleep(0.1)

        ticker = asyncio.ensure_future(tick())
        scope = {
            "type": "http",
            "headers": [(b"x-test-stamp", f"{arrival!r} {seq}".encode())],
        }
        await HttpChannel(app, _stamp)(scope, receive, None)
        await ticker

    thread = threading.Thread(target=deliver, name="body", daemon=True)
    thread.start()
    try:
        loop.run_until_complete(go())
        loop.run_forever()
    finally:
        thread.join(10)
        traffic.join(10)
    loop.close()
    assert handled == [(arrival, b"ab")]
    assert run.conn.unread == []


def test_requests_with_one_arrival_are_handed_over_in_seq_order(run):
    stamps, handled = queue.Queue(), []
    traffic = _traffic(run, stamps, before_ner=lambda: None, n=2)
    (arrival, _), (again, _) = stamps.get(timeout=10), stamps.get(timeout=10)
    assert arrival == again

    async def app(scope, receive, send):
        handled.append(_stamp(scope)[1])

    def read(seq):
        scope = {
            "type": "http",
            "headers": [(b"x-test-stamp", b"%r %d" % (arrival, seq))],
        }
        return HttpChannel(app, _stamp)(scope, _no_body, None)

    async def go():
        await asyncio.sleep(arrival)  # both are released, neither is read yet
        first = asyncio.ensure_future(read(1))
        for _ in range(3):  # the loop selects with seq 0 still unread
            await asyncio.sleep(0)
        await asyncio.gather(first, read(0))

    loop = CompassEventLoop()
    try:
        loop.run_until_complete(go())
        loop.run_forever()
    finally:
        traffic.join(10)
    loop.close()
    assert handled == [0, 1]
    assert run.conn.grants == [arrival, INF] and run.conn.unread == []


def test_requests_with_one_arrival_reach_the_app_in_seq_order_whatever_body_ends_first(
    run,
):
    stamps, entered = queue.Queue(), []
    traffic = _traffic(run, stamps, before_ner=lambda: None, n=2)
    (arrival, _), (again, _) = stamps.get(timeout=10), stamps.get(timeout=10)
    assert arrival == again

    async def app(scope, receive, send):
        entered.append(_stamp(scope)[1])

    def read(seq):
        async def receive():  # seq 0's body takes more socket reads than seq 1's
            for _ in range(3 if seq == 0 else 0):
                await asyncio.sleep(0)
            return {"type": "http.request", "body": b"x", "more_body": False}

        scope = {
            "type": "http",
            "headers": [(b"x-test-stamp", b"%r %d" % (arrival, seq))],
        }
        return HttpChannel(app, _stamp)(scope, receive, None)

    async def go():
        await asyncio.sleep(arrival)
        await asyncio.gather(read(0), read(1))

    loop = CompassEventLoop()
    try:
        loop.run_until_complete(go())
        loop.run_forever()
    finally:
        traffic.join(10)
    loop.close()
    print(f"\n  app entered in seq order {entered}")
    assert entered == [0, 1]
    assert run.conn.grants == [arrival, INF] and run.conn.unread == []


@pytest.mark.parametrize("order", [1, -1], ids=["entry-last", "entry-first"])
def test_a_stamp_split_over_two_tracestate_lines_holds_the_served_app(
    run, monkeypatch, order
):
    stamps, handled = queue.Queue(), []
    traffic = _traffic(
        run, stamps, before_ner=lambda: _until(lambda: run.rt.arrived[HTTP])
    )
    arrival, seq = stamps.get(timeout=10)

    async def app(scope, receive, send):
        handled.append(asyncio.get_running_loop().time())

    monkeypatch.setattr(api_server, "app", app)
    lines = [b"vendor=x", carriers.tracestate_with(None, arrival, seq).encode()]
    scope = {"type": "http", "headers": [(b"tracestate", v) for v in lines[::order]]}
    loop = CompassEventLoop()
    try:
        loop.run_until_complete(api_server._served_app()(scope, _no_body, None))
        assert handled == [arrival]
        loop.run_forever()
    finally:
        traffic.join(10)
    loop.close()
    assert run.conn.grants == [arrival, INF] and run.conn.unread == []


def test_compass_off_keeps_uvloop_and_on_names_a_loop_uvicorn_builds(run):
    clock.install(None)
    off = "uvloop" if importlib.util.find_spec("uvloop") else "auto"
    assert api_server._loop_impl() == off
    assert api_server._served_app() is api_server.app
    clock.install(run.rt)
    config = uvicorn.Config(lambda *_: None, loop=api_server._loop_impl())
    assert config.get_loop_factory() is CompassEventLoop


def test_select_returns_nothing_only_once_its_timeout_has_passed_on_the_lp_clock(
    run, monkeypatch
):
    _idle_traffic(run)
    calls, select = [], CompassSelector.select

    def recorded(self, timeout=None):
        t0 = self.loop.time()
        got = select(self, timeout)
        calls.append((t0, timeout, got, self.loop.time()))
        return got

    monkeypatch.setattr(CompassSelector, "select", recorded)
    loop = CompassEventLoop()

    async def go():
        later = asyncio.ensure_future(asyncio.sleep(1))

        def do_preprocess():
            time.sleep(0.2)  # open across the loop's next select

        await loop.run_in_executor(None, do_preprocess)
        await later

    loop.run_until_complete(go())
    loop.run_forever()
    loop.close()
    early = [
        c for c in calls if not c[2] and c[3] < c[0] + (INF if c[1] is None else c[1])
    ]
    assert calls and early == []


def test_a_stall_warns_once_across_diag_s_periods_and_unrelated_fd_events(run, caplog):
    _idle_traffic(run)
    run.rt.diag_s = 0.05
    loop = CompassEventLoop()
    stop = threading.Event()

    def poke():  # an unrelated fd event every 10 ms: a self-pipe write
        while not stop.wait(0.01):
            loop.call_soon_threadsafe(lambda: None)

    async def go():
        def do_preprocess():
            time.sleep(0.5)  # open for ten diag_s periods

        poker = threading.Thread(target=poke, daemon=True)
        poker.start()
        try:
            await loop.run_in_executor(None, do_preprocess)
        finally:
            stop.set()
            poker.join()

    with caplog.at_level(logging.WARNING, logger="atom"):
        loop.run_until_complete(go())
    loop.run_forever()
    loop.close()
    warned = [
        r.getMessage() for r in caplog.records if "still waiting" in r.getMessage()
    ]
    assert len(warned) == 1 and "station job 0 is open" in warned[0]


def test_http_channel_refuses_a_plain_asyncio_loop():
    scope = {"type": "http", "headers": [(b"x-test-stamp", b"0.5 0")]}
    with pytest.raises(TypeError, match="on a CompassEventLoop, not on"):
        asyncio.run(HttpChannel(None, _stamp)(scope, None, None))


CONTROL = "frontend->engine:control#dp0"


def _frontend_on_tcp(monkeypatch, poke, main):
    """Runs `main(loop)` on a frontend whose clock connection is tcp, so its
    loop serves its sockets while its next-event request is out.

    The frontend has an essential timer at 1. Traffic idles from the start; the
    engine joins only once `poke(loop)`, called while the frontend's request is
    out, has returned, so no grant reaches the frontend before then. Returns
    the frontend's runtime, the engine's control releases, and the loop's LP
    time at the finish.
    """
    authority = clock_transport.serve(ClockAuthority(TABLE), "tcp://127.0.0.1:0")
    # Kept referenced: a dropped connection closes its socket, ending the run.
    traffic = clock_transport.connect(TRAFFIC, authority.endpoint)
    traffic.send((NER, INF, [], INF))
    engine = clock_transport.connect(ENGINE, authority.endpoint)
    rt = LPRuntime(
        FRONTEND, TABLE, clock_transport.connect(FRONTEND, authority.endpoint)
    )
    rt.start_run()
    monkeypatch.setattr(clock, "_installed", rt)
    loop = CompassEventLoop()
    loop.call_at(1.0, lambda: None)
    released = []

    def engine_lp():
        _until(lambda: rt.asked)
        poke(loop)
        g = None
        while g != INF:
            engine.send((NER, INF, [], INF))
            g, got = engine.recv()
            released.extend(got[CONTROL])

    thread = threading.Thread(target=engine_lp, daemon=True)
    thread.start()
    # A loop the finish does not stop is stopped after 20 wall seconds.
    guard = threading.Timer(20, lambda: loop.call_soon_threadsafe(loop.stop))
    guard.start()
    try:
        main(loop)
        thread.join(10)
        return rt, released, loop.ended and loop.ended[0]
    finally:
        guard.cancel()
        authority.close()
        if loop.ended:
            loop.close()


def test_a_send_while_the_request_is_out_is_refused_naming_its_channel(monkeypatch):
    """A client hangs up while the frontend's request is out, and the handler
    stamps a control send, as the frontend's abort of that request does. The
    send is refused and the run goes on to its finish."""
    client, peer = socket.socketpair()
    refused, handling = [], threading.Event()

    def main(loop):
        def hang_up():
            loop.remove_reader(peer)
            handling.set()
            try:
                loop.rt.stamp_send(CONTROL)
            except RuntimeError as e:
                refused.append(str(e))

        loop.add_reader(peer, hang_up)
        loop.run_forever()

    def poke(loop):
        client.send(b"x")
        handling.wait(10)

    try:
        rt, released, end = _frontend_on_tcp(monkeypatch, poke, main)
    finally:
        client.close()
        peer.close()
    assert len(refused) == 1 and CONTROL in refused[0]
    assert rt.refusals == [
        f"clock:stamp_send on {CONTROL} while a next-event request is out"
    ]
    assert released == [] and end == 1.0


def test_a_keep_alive_timer_set_while_the_request_is_out_holds_no_finish(
    monkeypatch,
):
    """An unstamped GET is served while the frontend's request is out, as
    ``/health`` is before the traffic LP joins, and its connection stays open,
    so uvicorn's keep-alive timer is set. The run still finishes at the
    frontend's essential timer."""
    served, got = [], {}

    async def app(scope, receive, send):
        served.append(clock.installed().asked)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    server = uvicorn.Server(
        uvicorn.Config(
            app, host="127.0.0.1", port=0, lifespan="off", log_level="warning"
        )
    )

    def poke(loop):
        _until(lambda: server.started)
        port = server.servers[0].sockets[0].getsockname()[1]
        got["conn"] = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        got["conn"].request("GET", "/health")
        got["body"] = got["conn"].getresponse().read()

    try:
        _, _, end = _frontend_on_tcp(
            monkeypatch, poke, lambda loop: loop.run_until_complete(server.serve())
        )
    finally:
        if "conn" in got:
            got["conn"].close()
    assert got["body"] == b"ok" and served == [True]
    # The timer at 1 fires at the server tick that reaches it within asyncio's
    # clock resolution; the keep-alive timer, due at 5, never moves the clock.
    assert end == pytest.approx(1.0)


def test_a_stamped_cancel_hangs_its_request_up_at_its_arrival_never_the_socket(run):
    """Request 0 is stamped at 0 and its socket closes as soon as its body is
    read; the cancel naming it is stamped at 1. The app serving request 0
    reads ``http.disconnect`` at the cancel's arrival, and the cancel reaches
    no app: it is answered 204."""
    stamps = queue.Queue()

    def traffic():
        rt = LPRuntime(TRAFFIC, TABLE, clock_transport.connect(TRAFFIC, run.endpoint))
        rt.start_run()
        stamps.put(rt.stamp_send(HTTP))
        rt.advance_to(1.0)
        stamps.put(rt.stamp_send(HTTP))
        rt.next_event(INF)
        rt.close()

    thread = threading.Thread(target=traffic, name="traffic", daemon=True)
    thread.start()
    arrival, seq = TABLE.lookahead(HTTP), 0
    cancel_at, cancel = 1.0 + arrival, 1
    loop = CompassEventLoop()
    served, answered = [], []

    async def app(scope, receive, send):
        served.append((await receive())["body"])
        served.append(((await receive())["type"], loop.time()))

    def wire(body: bytes):
        parts = [{"type": "http.request", "body": body, "more_body": False}]

        async def receive():
            return parts.pop(0) if parts else {"type": "http.disconnect"}

        return receive

    async def answer(message):
        answered.append(message)

    def scope(a, s):
        return {"type": "http", "headers": [(b"x-test-stamp", b"%r %d" % (a, s))]}

    async def go():
        channel = HttpChannel(app, _stamp)
        await asyncio.gather(
            channel(scope(arrival, seq), wire(b"{}"), None),
            channel(scope(cancel_at, cancel), wire(carriers.cancel_body(seq)), answer),
        )

    try:
        loop.run_until_complete(go())
        loop.run_forever()
    finally:
        thread.join(10)
    loop.close()
    assert [stamps.get(timeout=10) for _ in range(2)] == [
        (arrival, seq),
        (cancel_at, cancel),
    ]
    assert served == [b"{}", ("http.disconnect", cancel_at)]
    assert answered == [
        {"type": "http.response.start", "status": 204},
        {"type": "http.response.body", "body": b""},
    ]


def _intake_run_file(monkeypatch, tmp_path, intake) -> None:
    """A run file whose machine spec states `intake` as host.http_intake, or
    with no machine spec for None."""
    from test_memory_readings import DOCUMENT

    from atom.compass import run as compass_run

    run = {"bound_s": 600.0}
    if intake is not None:
        machine = run["machine"] = copy.deepcopy(DOCUMENT)
        machine["host"]["http_intake"] = dict(zip(("fixed_s", "per_byte_s"), intake))
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(path))


def _burst(run, monkeypatch, n, app, sizes=None):
    """`n` stamped requests at one arrival, through the served app; returns
    the arrival once the loop has finished."""
    stamps = queue.Queue()
    traffic = _traffic(run, stamps, before_ner=lambda: None, n=n)
    arrival = stamps.get(timeout=10)[0]
    for _ in range(n - 1):
        assert stamps.get(timeout=10)[0] == arrival
    monkeypatch.setattr(api_server, "app", app)
    served = api_server._served_app()

    def read(seq):
        async def receive():
            body = b"x" * (sizes[seq] if sizes else 0)
            return {"type": "http.request", "body": body}

        tracestate = carriers.tracestate_with(None, arrival, seq).encode()
        scope = {"type": "http", "headers": [(b"tracestate", tracestate)]}
        return served(scope, receive, None)

    async def go():
        await asyncio.sleep(arrival)
        await asyncio.gather(*(read(seq) for seq in range(n)))

    loop = CompassEventLoop()
    try:
        loop.run_until_complete(go())
        loop.run_forever()
    finally:
        traffic.join(10)
    loop.close()
    return arrival


@pytest.mark.parametrize("intake", [(2e-3, 1e-6), None], ids=["probed", "unstated"])
def test_a_burst_reaches_the_app_one_intake_charge_at_a_time(
    run, monkeypatch, tmp_path, intake
):
    _intake_run_file(monkeypatch, tmp_path, intake)
    sizes, entered = {0: 1000, 1: 3000}, []

    async def app(scope, receive, send):
        entered.append(asyncio.get_running_loop().time())

    arrival = _burst(run, monkeypatch, 2, app, sizes)
    fixed, per_byte = intake or (0.0, 0.0)
    first = arrival + fixed + sizes[0] * per_byte
    print(f"\n  app entered at {entered}")
    assert entered == pytest.approx([first, first + fixed + sizes[1] * per_byte])


def test_an_intake_waits_for_the_open_tokenizer_job_so_its_result_is_not_late(
    run, monkeypatch, tmp_path
):
    # Each intake costs 2 ms; each request's tokenizer job is charged 1 ms, and
    # the first one's real thread takes 0.3 wall seconds. Job 0 completes at
    # 3 ms, inside request 1's intake, so its result runs when that intake
    # ends, at 4 ms, not once the clock has run on through every intake.
    _intake_run_file(monkeypatch, tmp_path, (2e-3, 1e-9))  # empty bodies
    entry = SimpleNamespace(encode_fixed_s=1e-3, encode_tokens_per_s=1.0, derate=1.0)
    encode, done = wrap_encode(list, entry), []

    async def app(scope, receive, send):
        seq = api_server._tracestate_stamp(scope)[1]

        def do_preprocess():
            time.sleep(0.3 if seq == 0 else 0)
            encode()

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, do_preprocess)
        done.append((seq, loop.time()))

    arrival = _burst(run, monkeypatch, 3, app)
    print(f"\n  tokenizer results at {done}")
    expected = [(0, 4e-3), (1, 6e-3), (2, 7e-3)]
    assert [s for s, _ in done] == [s for s, _ in expected]
    assert [t for _, t in done] == pytest.approx([arrival + t for _, t in expected])


def test_a_tokenizer_result_due_when_an_intake_starts_runs_before_its_charge(
    run, monkeypatch, tmp_path
):
    # Each intake costs 2 ms and each tokenizer job is charged nothing; the
    # first one's real thread takes 0.3 wall seconds. Job 0 completes at 2 ms,
    # the instant request 1's intake starts, so its result runs then, before
    # that intake is charged, not when the intake ends at 4 ms.
    _intake_run_file(monkeypatch, tmp_path, (2e-3, 1e-9))  # empty bodies
    entry = SimpleNamespace(encode_fixed_s=0.0, encode_tokens_per_s=1.0, derate=1.0)
    encode, done = wrap_encode(list, entry), []

    async def app(scope, receive, send):
        seq = api_server._tracestate_stamp(scope)[1]

        def do_preprocess():
            time.sleep(0.3 if seq == 0 else 0)
            encode()

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, do_preprocess)
        done.append((seq, loop.time()))

    arrival = _burst(run, monkeypatch, 2, app)
    print(f"\n  tokenizer results at {done}")
    assert [t for _, t in sorted(done)] == pytest.approx(
        [arrival + 2e-3, arrival + 4e-3]
    )
