# SPDX-License-Identifier: MIT
"""`CompassEventLoop` against the co-hosted clock authority.

Three LPs as in a one-engine run. The frontend is the loop under test; the
engine waits in NER(+inf), and so does the traffic LP unless a test drives it.
The daemon timers are the real ones: the API server's metrics refresh loop,
and uvicorn's server tick and keep-alive timeout.
"""

import asyncio
import http.client
import importlib.util
import logging
import math
import queue
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
from atom.utils.compass_loop import CompassEventLoop, CompassSelector, HttpChannel

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


def test_the_metrics_refresh_is_a_daemon_and_the_finish_stops_the_loop(
    run, monkeypatch
):
    _idle_traffic(run)
    refreshed, late = [], []

    async def refresh():
        loop = asyncio.get_running_loop()
        refreshed.append(loop.time())
        # A plain callback, a daemon because the refresh loop schedules it.
        loop.call_later(4, lambda: late.append(loop.time()))

    monkeypatch.setattr(api_server, "_refresh_metrics_once", refresh)

    async def serve():
        asyncio.create_task(api_server._metrics_refresh_loop())
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
        assert (await receive())["type"] == "http.disconnect"

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
