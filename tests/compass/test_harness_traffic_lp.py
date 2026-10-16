# SPDX-License-Identifier: MIT
"""compass_harness's traffic LP against the Clock Authority.

The trace tests serve a Clock Authority in this process with an idle engine
LP and a stand-in frontend LP. The frontend thread also stands in for the
aiperf worker: it reads the published stamps, reports each stream event it
sends and hands each final event's credit return to the router. aiperf's
strategy is a stand-in that schedules the next turn a fixed delay after a
return, on a ``ClockPacedLoopScheduler`` paced by the traffic LP. The hold
tests replace the authority with a scripted runtime, so a return can be put
on either side of its final event's release. Skips by name as
``test_harness_pacing.py`` does.
"""

import asyncio
import heapq
import json
import math
import re
import shutil
import threading
import uuid
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest


def _missing() -> str | None:
    try:
        metadata.version("compass-harness")
        import compass_harness  # noqa: F401  checks aiperf against its pinned source
    except metadata.PackageNotFoundError as e:
        return f"{e.name} is not installed"
    except RuntimeError as e:
        return str(e)
    return None


if _why := _missing():
    pytest.skip(
        f"needs agentx-harness 56a0cf70 and compass-harness: {_why}",
        allow_module_level=True,
    )

from aiperf.plugin import plugins  # noqa: F401  plugin discovery, as in aiperf

# isort: split
import aiperf
import zmq
import zmq.asyncio
from aiperf.common.enums import CreditPhase
from aiperf.credit.sticky_router import StickyCreditRouter
from compass_harness.router import CompassCreditRouter
from compass_harness.scheduler import ClockPacedLoopScheduler
from compass_harness.traffic_lp import (
    HTTP,
    STREAM,
    TrafficLP,
    UnansweredRequests,
)
from compass_harness.transport import addresses, credit_key

from atom.compass import clock_transport
from atom.compass.clock import (
    NER,
    ClockAuthority,
    LpId,
    prefill_decode_table,
    single_engine_table,
)
from atom.compass.run import ENV as RUN_ENV
from atom.utils.clock import LPRuntime, Straggler
from compass_harness import ADDRESS_ENV, fingerprint

#: The run file's table keys, as the deployment writes them.
RUN = {"admission_path": "serving", "ipc_s": 2.0**-10, "stream_s": 2.0**-9}
TABLE = single_engine_table(**RUN)
FRONTEND, ENGINE = LpId("frontend"), LpId("engine")
INF = math.inf
ADMIT = TABLE.lookahead(HTTP)
STREAM_S = TABLE.lookahead(STREAM)
#: Each request's stream events, as offsets from its admission; the last is final.
EVENTS = (0.5, 1.0)
DELAY = 0.25  # the stand-in strategy's gap between a return and the next turn


def _credit(num: int):
    return SimpleNamespace(id=num, phase=CreditPhase.PROFILING, phase_index=0)


def _key(num: int) -> tuple:
    return credit_key(CreditPhase.PROFILING, 0, num)


class Frontend(threading.Thread):
    """The frontend LP, and the worker that carries its stream to the traffic LP."""

    def __init__(self, endpoint, prefix, loop, held, drop=()) -> None:
        super().__init__(name="frontend", daemon=True)
        self.endpoint, self.loop, self.held, self.drop = endpoint, loop, held, drop
        self.stamps, self.reports = addresses(prefix)
        self.rt = None

    def run(self) -> None:
        self.rt = rt = LPRuntime(
            FRONTEND, TABLE, clock_transport.connect(FRONTEND, self.endpoint)
        )
        rt.start_run()
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.connect(self.stamps)
        sub.subscribe(b"")
        push = ctx.socket(zmq.PUSH)
        push.connect(self.reports)
        keys, handled, due = {}, set(), []
        while True:
            for seq in sorted(rt.released[HTTP] - handled):
                while seq not in keys:
                    key, _, stamped = sub.recv_pyobj()
                    keys[stamped] = key
                handled.add(seq)
                for i, offset in enumerate(EVENTS):
                    final = i == len(EVENTS) - 1
                    if not (final and keys[seq] in self.drop):
                        heapq.heappush(due, (rt.now + offset, keys[seq], final))
            while due and due[0][0] <= rt.now:
                _, key, final = heapq.heappop(due)
                arrival, seq = rt.stamp_send(STREAM)
                push.send_pyobj((key, seq, arrival, final))
                if final:
                    ret = SimpleNamespace(credit=_credit(key[2]), error=None)
                    self.loop.call_soon_threadsafe(
                        asyncio.ensure_future, self.held("worker_0", ret)
                    )
            if rt.next_event(due[0][0] if due else INF) == INF:
                rt.close()
                ctx.destroy(linger=0)
                return


async def _trace(endpoint, prefix, monkeypatch, *, requests, drop=(), logged=None):
    """Run `requests` turns, each `DELAY` after the last one's return."""
    monkeypatch.setattr(StickyCreditRouter, "__init__", lambda self, **kw: None)
    sent = []
    monkeypatch.setattr(
        StickyCreditRouter, "send_credit", lambda self, credit: _record(sent, credit)
    )
    logged = [] if logged is None else logged
    monkeypatch.setattr(
        CompassCreditRouter, "exception", lambda self, msg: logged.append(msg)
    )
    router = CompassCreditRouter(run=None, service_id="timing_manager")
    router._workers = {"worker_0": None}
    traffic = router.traffic
    scheduler = ClockPacedLoopScheduler()
    times = {}

    async def issue(num):
        times[num] = {"sent": traffic.now()}
        await router.send_credit(_credit(num))

    async def on_return(worker_id, ret):
        num = ret.credit.id
        times[num]["returned"] = traffic.now()
        if num + 1 < requests:
            scheduler.schedule_later(DELAY, issue(num + 1))
        else:
            router.mark_credits_complete()

    router.set_return_callback(on_return)
    frontend = Frontend(
        endpoint, prefix, asyncio.get_running_loop(), router._on_return_callback, drop
    )
    frontend.start()
    await router.wait_for_workers(5)
    await issue(0)
    try:
        # Only the router stops the loop when the finish names an open request.
        await (asyncio.Event().wait() if drop else traffic.done)
    finally:
        frontend.join(10)
    return SimpleNamespace(
        traffic=traffic, frontend=frontend, times=times, sent=sent, logged=logged
    )


async def _record(sent, credit) -> None:
    sent.append(credit.id)


@pytest.fixture
def ca(tmp_path, monkeypatch):
    endpoint = f"inproc:test-traffic-lp-{uuid.uuid4().hex}"
    server = clock_transport.serve(ClockAuthority(TABLE), endpoint)
    clock_transport.connect(ENGINE, endpoint).send((NER, INF, [], INF))
    run_file = tmp_path / "run.json"
    run_file.write_text(json.dumps({"clock_endpoint": endpoint, "bound_s": 1e6, **RUN}))
    monkeypatch.setenv(RUN_ENV, str(run_file))
    monkeypatch.setenv(ADDRESS_ENV, str(tmp_path / "lp"))
    monkeypatch.setattr(ClockPacedLoopScheduler, "clock", None)
    try:
        yield SimpleNamespace(endpoint=endpoint, prefix=str(tmp_path / "lp"))
    finally:
        server.close()


def _run(coro):
    """Run `coro`, failing rather than hanging when the run never finishes."""
    return asyncio.run(asyncio.wait_for(coro, 30))


def test_a_three_request_trace_ends_by_the_finish(ca, monkeypatch):
    run = _run(_trace(ca.endpoint, ca.prefix, monkeypatch, requests=3))
    assert run.traffic.rt.now == INF and run.frontend.rt.now == INF
    assert run.sent == [0, 1, 2] and run.logged == []
    print("\nrequest  sent    first token  final   next turn")
    sent = 0.0
    for num in range(3):
        admitted = sent + ADMIT
        first = admitted + EVENTS[0] + STREAM_S
        final = admitted + EVENTS[-1] + STREAM_S
        assert run.times[num] == {"sent": sent, "returned": final}
        print(f"{num:7d} {sent:7.4f} {first:12.4f} {final:7.4f} {final + DELAY:10.4f}")
        sent = final + DELAY


def test_a_dropped_response_is_named_at_the_finish(ca, monkeypatch):
    logged = []
    named = re.escape(
        "the run finished with requests [('profiling', 0, 1)] sent but never given "
        "their final response"
    )
    with pytest.raises(RuntimeError, match="Event loop stopped before Future"):
        _run(
            _trace(
                ca.endpoint,
                ca.prefix,
                monkeypatch,
                requests=3,
                drop={_key(1)},
                logged=logged,
            )
        )
    # The router logged it and stopped the loop, which fails the timing manager.
    (message,) = logged
    assert re.search(named, message)


class ScriptedRuntime:
    """The traffic LP's runtime with its grants scripted instead of asked for.

    Each script entry ``(G, seqs)`` is the next grant and the stream seqs it
    releases; a clock call whose target comes first is granted the target.
    """

    def __init__(self, script, stream=STREAM) -> None:
        self.now, self.script, self.seq = 0.0, list(script), 0
        self.stream, self.released = stream, {stream: set()}
        self.calls = []

    def stamp_send(self, ch):
        self.seq += 1
        return self.now + ADMIT, self.seq - 1

    def check_arrival(self, ch, arrival, seq) -> None:
        pass

    def next_event(self, t):
        self.calls.append(t)
        if self.script and self.script[0][0] <= t:
            self.now, seqs = self.script.pop(0)
            self.released[self.stream] |= seqs
        else:
            self.now = t
        return self.now

    def close(self) -> None:
        pass


@pytest.mark.parametrize("order", ["before", "after"])
def test_a_return_before_or_after_its_final_release_schedules_from_the_final(
    order, tmp_path, monkeypatch
):
    """Credit 0's final event, stream seq 0, arrives at 1.0; its return is
    passed to aiperf at 1.0 whichever side of the release it reaches the hold.
    Credit 9 stays open throughout, so only the wait for aiperf to block keeps
    the owner from asking for time before the next turn is scheduled."""

    async def main():
        rt = ScriptedRuntime([(1.0, {0})])
        traffic = TrafficLP(rt, str(tmp_path / "lp"))
        monkeypatch.setattr(ClockPacedLoopScheduler, "clock", traffic)
        scheduler = ClockPacedLoopScheduler()
        sends = {}

        async def issue(num):
            sends[num] = traffic.now()
            traffic.send(_key(num))

        async def on_return(worker_id, ret):
            scheduler.schedule_later(DELAY, issue(ret.credit.id + 1))

        held = traffic.hold(on_return)
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(addresses(str(tmp_path / "lp"))[1])
        traffic.send(_key(9))
        await issue(0)
        if order == "after":
            while rt.now < 1.0:  # the release
                await asyncio.sleep(0.001)
        push.send_pyobj((_key(0), 0, 1.0, True))
        if order == "after":
            while 0 not in traffic._reports:  # the owner holds for the return
                await asyncio.sleep(0.001)
            await asyncio.sleep(0.01)
        await held("worker_0", SimpleNamespace(credit=_credit(0), error=None))
        while 1 not in sends:
            await asyncio.sleep(0.001)
        push.close(linger=0)
        if traffic.done.done():
            traffic.done.exception()  # UnansweredRequests: 9 and 1 are open
        else:
            traffic.done.cancel()
        return sends, rt.calls

    sends, calls = _run(main())
    assert sends == {0: 0.0, 1: 1.0 + DELAY}
    # Nothing asks for time between the release and the return.
    assert calls[:2] == [INF, 1.0 + DELAY]


@pytest.mark.parametrize("order", ["before", "after"])
def test_an_errored_return_counts_its_released_final_as_returned(order, tmp_path):
    """aiohttp can raise after ``[DONE]``: the return carries an error although
    the final event was reported. Either side of the release, it is passed once
    and the run reaches the finish."""

    async def main():
        rt = ScriptedRuntime([(1.0, {0})])
        traffic = TrafficLP(rt, str(tmp_path / "lp"))
        passed = []

        async def on_return(worker_id, ret):
            passed.append(ret.credit.id)

        held = traffic.hold(on_return)
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(addresses(str(tmp_path / "lp"))[1])
        traffic.send(_key(0))
        push.send_pyobj((_key(0), 0, 1.0, True))
        if order == "after":
            while rt.now < 1.0 or 0 not in traffic._reports:
                await asyncio.sleep(0.001)
        error = SimpleNamespace(credit=_credit(0), error="ClientPayloadError")
        await held("worker_0", error)
        traffic.finish()
        await asyncio.wait_for(traffic.done, 5)
        push.close(linger=0)
        return passed, rt.calls

    passed, calls = _run(main())
    assert passed == [0] and calls == [INF, INF]


def test_a_request_failing_mid_stream_fails_the_hold_by_name(tmp_path, monkeypatch):
    """Credit 0's worker reported seq 0, then its request failed: the released
    final, seq 1, is never reported."""
    monkeypatch.setattr("compass_harness.traffic_lp.DIAG_S", 0.05)

    async def main():
        rt = ScriptedRuntime([(1.0, {0, 1})])
        traffic = TrafficLP(rt, str(tmp_path / "lp"))

        async def on_return(worker_id, ret):
            pass

        held = traffic.hold(on_return)
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(addresses(str(tmp_path / "lp"))[1])
        traffic.send(_key(0))
        push.send_pyobj((_key(0), 0, 1.0, False))
        await held("worker_0", SimpleNamespace(credit=_credit(0), error="TimeoutError"))
        traffic.finish()
        try:
            await asyncio.wait_for(traffic.done, 5)
        finally:
            push.close(linger=0)

    with pytest.raises(UnansweredRequests) as failed:
        _run(main())
    print(f"\n{failed.value}")
    assert str(failed.value) == (
        "requests [('profiling', 0, 0)] failed mid-stream: released stream seqs [1] "
        "were not reported after 0.05 wall seconds"
    )


def test_a_slow_report_with_no_errored_request_keeps_the_hold_waiting(
    tmp_path, monkeypatch
):
    """Credit 0's worker reports its released final, seq 0, six ``DIAG_S``
    periods late and its request did not error: the hold waits for it."""
    monkeypatch.setattr("compass_harness.traffic_lp.DIAG_S", 0.05)

    async def main():
        rt = ScriptedRuntime([(1.0, {0})])
        traffic = TrafficLP(rt, str(tmp_path / "lp"))
        passed = []

        async def on_return(worker_id, ret):
            passed.append(ret.error)

        held = traffic.hold(on_return)
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(addresses(str(tmp_path / "lp"))[1])
        traffic.send(_key(0))
        await asyncio.sleep(0.3)
        push.send_pyobj((_key(0), 0, 1.0, True))
        await held("worker_0", SimpleNamespace(credit=_credit(0), error=None))
        traffic.finish()
        try:
            await asyncio.wait_for(traffic.done, 5)
        finally:
            push.close(linger=0)
        return passed, rt.calls

    passed, calls = _run(main())
    assert passed == [None] and calls[-1] == INF


class ScriptedConn:
    """A Clock Authority connection whose grants are scripted: grant ``k``
    releases stream seq ``k`` at ``k + 1``, then ``+inf`` ends the run. Each
    clock call records how many released stream seqs the runtime still holds."""

    def __init__(self, grants: int) -> None:
        self.rt, self.grants, self.k, self.held_at_call = None, grants, 0, []

    def send(self, msg) -> None:
        self.held_at_call.append(len(self.rt.released[STREAM]))

    def recv(self):
        if self.k == self.grants:
            return INF, {}
        self.k += 1
        return float(self.k), {STREAM: [(self.k - 1, float(self.k))]}


def test_a_pass_scans_only_the_open_stream_events(tmp_path):
    """Many requests, each with one final stream event, run through a real
    runtime's grants: at every clock call the runtime holds no released event
    the traffic LP has already handled."""
    n = 200

    async def main():
        conn = ScriptedConn(n)
        conn.rt = rt = LPRuntime(LpId("traffic"), TABLE, conn)
        traffic = TrafficLP(rt, str(tmp_path / "lp"))
        passed = []

        async def on_return(worker_id, ret):
            passed.append(ret.credit.id)

        held = traffic.hold(on_return)
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(addresses(str(tmp_path / "lp"))[1])
        for num in range(n):
            traffic.send(_key(num))
            await held("worker_0", SimpleNamespace(credit=_credit(num), error=None))
            push.send_pyobj((_key(num), num, float(num + 1), True))
        traffic.finish()
        try:
            await asyncio.wait_for(traffic.done, 10)
        finally:
            push.close(linger=0)
        return passed, conn.held_at_call, rt

    passed, held_at_call, rt = _run(main())
    assert passed == list(range(n)) and rt.now == INF
    assert held_at_call == [0] * (n + 1)


def test_a_raising_return_callback_does_not_stop_later_returns(tmp_path, caplog):
    async def main():
        traffic = TrafficLP(ScriptedRuntime([]), str(tmp_path / "lp"))
        passed = []

        async def on_return(worker_id, ret):
            passed.append(ret.credit.id)
            if ret.credit.id == 0:
                raise RuntimeError("boom")

        held = traffic.hold(on_return)
        for num in (0, 1):
            await held("w", SimpleNamespace(credit=_credit(num), error="refused"))
        while len(passed) < 2:
            await asyncio.sleep(0.001)
        traffic.done.cancel()
        return passed

    assert _run(main()) == [0, 1]
    assert "a return callback raised" in caplog.text


def test_a_second_send_credit_call_site_refuses_the_package(tmp_path):
    shutil.copytree(Path(aiperf.__file__).parent, tmp_path / "aiperf")
    fingerprint.check(tmp_path)
    path = tmp_path / "aiperf" / "timing" / "intervals.py"
    path.write_text(
        path.read_text() + "\n\nasync def _resend(router, credit):\n"
        "    await router.send_credit(credit=credit)\n"
    )
    with pytest.raises(
        RuntimeError, match=re.escape("2 .send_credit( call sites, not 1")
    ):
        fingerprint.check(tmp_path)


def test_the_timing_manager_builds_the_compass_router():
    from aiperf.timing import manager

    assert manager.StickyCreditRouter is CompassCreditRouter


def test_a_1p1d_run_file_sends_to_frontend_p_and_reads_from_frontend_d(
    tmp_path, monkeypatch
):
    """Behind a prefill-decode router, requests enter at frontend-P and the
    stream comes back from frontend-D, over the channels the run file's table
    declares."""
    pd = {"router_s": 2.0**-12, "kv_write_req_s": 2.0**-10}
    endpoint = f"inproc:test-traffic-lp-{uuid.uuid4().hex}"
    table = prefill_decode_table(**RUN, **pd)
    server = clock_transport.serve(ClockAuthority(table), endpoint)
    run_file = tmp_path / "run.json"
    run_file.write_text(
        json.dumps({"clock_endpoint": endpoint, "bound_s": 1e6, **RUN, **pd})
    )
    monkeypatch.setenv(RUN_ENV, str(run_file))
    monkeypatch.setenv(ADDRESS_ENV, str(tmp_path / "lp"))

    async def main():
        traffic = TrafficLP.from_env()
        try:
            traffic.send(_key(0))
            traffic.rt.check_arrival("frontend-D->traffic:stream", 1.0, 0)
            return traffic.rt.send_log, traffic.rt.unreleased
        finally:
            for task in (*traffic._tasks, traffic.done):
                task.cancel()
            traffic._pub.close(linger=0)
            traffic._pull.close(linger=0)

    try:
        sent, unreleased = _run(main())
    finally:
        server.close()
    http = "traffic->frontend-P:http"
    assert sent == [(http, 0, table.lookahead(http))]
    assert unreleased == {("frontend-D->traffic:stream", 0): 1.0}


def test_behind_the_router_lp_requests_and_the_stream_cross_it_and_cancels_stay_off(
    tmp_path, monkeypatch
):
    """With a router LP the traffic LP sends to it and reads the stream from it,
    and a cancel, which the router would refuse, is never stamped."""
    pd = {"router_s": 2.0**-12, "kv_write_req_s": 2.0**-10, "router_in_s": 2.0**-14}
    endpoint = f"inproc:test-traffic-lp-{uuid.uuid4().hex}"
    server = clock_transport.serve(
        ClockAuthority(prefill_decode_table(**RUN, **pd)), endpoint
    )
    run_file = tmp_path / "run.json"
    run_file.write_text(
        json.dumps({"clock_endpoint": endpoint, "bound_s": 1e6, **RUN, **pd})
    )
    monkeypatch.setenv(RUN_ENV, str(run_file))
    monkeypatch.setenv(ADDRESS_ENV, str(tmp_path / "lp"))

    async def main():
        traffic = TrafficLP.from_env()
        try:
            return traffic.http, traffic.stream, traffic.cancels
        finally:
            for task in (*traffic._tasks, traffic.done):
                task.cancel()
            traffic._pub.close(linger=0)
            traffic._pull.close(linger=0)

    try:
        got = _run(main())
    finally:
        server.close()
    assert got == ("traffic->router:http", "router->traffic:stream", False)


def test_a_1p1d_hold_settles_the_frontend_d_stream(tmp_path):
    """Behind a prefill-decode router the hold reads frontend-D's stream: the
    released final is reported, its return passed, and the run finishes."""
    stream = "frontend-D->traffic:stream"

    async def main():
        rt = ScriptedRuntime([(1.0, {0})], stream)
        traffic = TrafficLP(rt, str(tmp_path / "lp"), stream=stream)
        passed = []

        async def on_return(worker_id, ret):
            passed.append(ret.credit.id)

        held = traffic.hold(on_return)
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(addresses(str(tmp_path / "lp"))[1])
        traffic.send(_key(0))
        push.send_pyobj((_key(0), 0, 1.0, True))
        await held("worker_0", SimpleNamespace(credit=_credit(0), error=None))
        traffic.finish()
        try:
            await asyncio.wait_for(traffic.done, 5)
        finally:
            push.close(linger=0)
        return passed, rt.released

    passed, released = _run(main())
    assert passed == [0] and released == {stream: set()}


def test_a_cancel_closes_the_open_requests_and_drops_their_later_events(tmp_path):
    """Credits 0 and 1 are open at the grace deadline, 2.0. The cancel stamps
    one http send per credit for its worker; their cancelled returns, which
    the worker sends 50 ms after writing the cancels, are passed before the
    next clock call. Credit 0's event read past the cancel is recorded with no
    arrival check, and the event released after it that no worker reports is
    dropped, so the run finishes with nothing open."""

    class Runtime(ScriptedRuntime):
        def next_event(self, t):
            passed_at.append(list(passed))
            return super().next_event(t)

    passed, passed_at = [], []

    async def main():
        rt = Runtime([(1.0, {0}), (3.0, {1, 2})])
        traffic = TrafficLP(rt, str(tmp_path / "lp"))
        stamps, reports = addresses(str(tmp_path / "lp"))
        sub = zmq.asyncio.Context.instance().socket(zmq.SUB)
        sub.connect(stamps)
        sub.subscribe(b"")
        await traffic.subscribed(1)
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(reports)

        async def on_return(worker_id, ret):
            passed.append(ret.credit.id)

        held = traffic.hold(on_return)

        async def worker():
            published = [await sub.recv_pyobj() for _ in range(4)]
            for key, *_ in published[2:]:
                push.send_pyobj((key,))
            push.send_pyobj((_key(0), 1, 2.5, False))
            await asyncio.sleep(0.05)
            for num in (1, 0):
                await held("w", SimpleNamespace(credit=_credit(num), error=None))
            return published

        worker = asyncio.ensure_future(worker())
        traffic.send(_key(0))
        traffic.send(_key(1))
        push.send_pyobj((_key(0), 0, 1.0, False))
        while traffic.now() < 2.0:
            await traffic.wait(2.0)
        await traffic.cancel()
        traffic.finish()
        try:
            await asyncio.wait_for(traffic.done, 5)
        finally:
            push.close(linger=0)
            sub.close(linger=0)
        return await worker, rt.calls

    published, calls = _run(main())
    assert published[2:] == [
        (_key(0), 2.0 + ADMIT, 2, "cancel"),
        (_key(1), 2.0 + ADMIT, 3, "cancel"),
    ]
    assert calls == [2.0, 2.0, INF, INF]
    assert passed_at == [[], [], [0, 1], [0, 1]]


def test_returns_whose_finals_share_a_grant_pass_in_stream_order(tmp_path):
    """Credits 0 and 1 end in one grant, stream seqs 0 and 1. Credit 1's return
    reaches the hold first and credit 0's 50 ms later; both are passed in
    stream order, so the next turns aiperf schedules from them keep it."""

    async def main():
        rt = ScriptedRuntime([(1.0, {0, 1})])
        traffic = TrafficLP(rt, str(tmp_path / "lp"))
        passed = []

        async def on_return(worker_id, ret):
            passed.append(ret.credit.id)

        held = traffic.hold(on_return)
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(addresses(str(tmp_path / "lp"))[1])
        traffic.send(_key(0))
        traffic.send(_key(1))
        push.send_pyobj((_key(0), 0, 1.0, True))
        push.send_pyobj((_key(1), 1, 1.0, True))
        await held("w", SimpleNamespace(credit=_credit(1), error=None))
        while rt.now < 1.0 or 1 not in traffic._reports:
            await asyncio.sleep(0.001)
        await asyncio.sleep(0.05)
        await held("w", SimpleNamespace(credit=_credit(0), error=None))
        traffic.finish()
        try:
            await asyncio.wait_for(traffic.done, 5)
        finally:
            push.close(linger=0)
        return passed

    assert _run(main()) == [0, 1]


def test_a_cancel_waits_for_a_released_final_s_return_before_it_stamps(tmp_path):
    """Credits 0 and 1 are open; the grant at the grace deadline, 2.0, releases
    credit 0's final, whose return reaches the hold 50 ms after aiperf cancels.
    The cancel waits for that return, so credit 0 completes and only credit 1
    gets a cancel."""

    async def main():
        rt = ScriptedRuntime([(2.0, {0})])
        traffic = TrafficLP(rt, str(tmp_path / "lp"))
        stamps, reports = addresses(str(tmp_path / "lp"))
        sub = zmq.asyncio.Context.instance().socket(zmq.SUB)
        sub.connect(stamps)
        sub.subscribe(b"")
        await traffic.subscribed(1)
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(reports)
        passed, cancels = [], []

        async def on_return(worker_id, ret):
            passed.append(ret.credit.id)

        held = traffic.hold(on_return)

        async def worker():
            while True:
                key, *_, kind = await sub.recv_pyobj()
                if kind == "cancel":
                    cancels.append(key)
                    push.send_pyobj((key,))
                    ret = SimpleNamespace(credit=_credit(key[2]), error=None)
                    await held("w", ret)

        async def late_return():
            await asyncio.sleep(0.05)
            await held("w", SimpleNamespace(credit=_credit(0), error=None))

        worker = asyncio.ensure_future(worker())
        traffic.send(_key(0))
        traffic.send(_key(1))
        push.send_pyobj((_key(0), 0, 2.0, True))
        while traffic.now() < 2.0:
            await traffic.wait(2.0)
        late = asyncio.ensure_future(late_return())
        await traffic.cancel()
        traffic.finish()
        try:
            await asyncio.wait_for(traffic.done, 5)
        finally:
            worker.cancel()
            late.cancel()
            push.close(linger=0)
            sub.close(linger=0)
        return cancels, passed

    cancels, passed = _run(main())
    assert cancels == [_key(1)]
    assert passed == [0, 1]


class StragglerRuntime(ScriptedRuntime):
    """``check_arrival`` with ``LPRuntime``'s rule: a report of an unreleased seq
    whose arrival the LP has already passed is a straggler."""

    def check_arrival(self, ch, arrival, seq) -> None:
        if seq not in self.released[ch] and arrival < self.now:
            raise Straggler(f"{ch} seq {seq} arrives at {arrival}, now {self.now}")


def test_a_cancelled_request_s_report_read_after_its_seq_is_dropped(tmp_path):
    """Credits 0 and 1 are cancelled at 2.0 and their returns passed. The grant
    at 3.0 releases credit 0's stream seq 1 with no request open, so the hold
    drops it; its report (arrival 2.5) is read after that. Credit 5, sent next,
    still has its return passed and the run finishes."""

    async def main():
        rt = StragglerRuntime([(1.0, {0}), (3.0, {1}), (4.0, {2})])
        traffic = TrafficLP(rt, str(tmp_path / "lp"))
        stamps, reports = addresses(str(tmp_path / "lp"))
        sub = zmq.asyncio.Context.instance().socket(zmq.SUB)
        sub.connect(stamps)
        sub.subscribe(b"")
        await traffic.subscribed(1)
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(reports)
        passed = []

        async def on_return(worker_id, ret):
            passed.append(ret.credit.id)

        held = traffic.hold(on_return)

        async def worker():
            published = [await sub.recv_pyobj() for _ in range(4)]
            for key, *_ in published[2:]:
                push.send_pyobj((key,))
            await asyncio.sleep(0.05)
            for num in (0, 1):
                await held("w", SimpleNamespace(credit=_credit(num), error=None))

        worker = asyncio.ensure_future(worker())
        traffic.send(_key(0))
        traffic.send(_key(1))
        push.send_pyobj((_key(0), 0, 1.0, False))
        while traffic.now() < 2.0:
            await traffic.wait(2.0)
        await traffic.cancel()
        await worker
        while traffic.now() < 3.0:
            await traffic.wait(3.0)
        push.send_pyobj((_key(0), 1, 2.5, False))
        await asyncio.sleep(0.1)
        traffic.send(_key(5))
        push.send_pyobj((_key(5), 2, 3.5, True))
        await held("w", SimpleNamespace(credit=_credit(5), error=None))
        traffic.finish()
        try:
            await asyncio.wait_for(traffic.done, 5)
        finally:
            push.close(linger=0)
            sub.close(linger=0)
        return passed

    assert _run(main()) == [0, 1, 5]


def test_a_report_the_reader_rejects_fails_the_run(tmp_path):
    """Credit 0 is open and its stream seq 0 released at 1.0, unreported. A
    report of seq 1 with arrival 0.5, read after that grant, is a straggler:
    the run fails with it instead of holding for seq 0 forever."""

    async def main():
        rt = StragglerRuntime([(1.0, {0})])
        traffic = TrafficLP(rt, str(tmp_path / "lp"))
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(addresses(str(tmp_path / "lp"))[1])
        traffic.send(_key(0))
        while traffic.now() < 1.0:
            await traffic.wait(1.0)
        push.send_pyobj((_key(0), 1, 0.5, False))
        try:
            await asyncio.wait_for(traffic.done, 5)
        finally:
            push.close(linger=0)

    with pytest.raises(Straggler, match="seq 1 arrives at 0.5, now 1.0"):
        _run(main())


def test_a_failed_traffic_lp_tells_aiperf_the_run_failed_before_it_stops(
    tmp_path, monkeypatch
):
    """The reader rejects a report, as above, under the router. Before the loop
    stops, the router publishes an error the controller keeps as an exit error,
    then the cancel the records manager finishes on."""
    from aiperf.common.messages import ProfileCancelCommand
    from aiperf.controller.result_join_coordinator import ResultJoinCoordinator
    from aiperf.controller.system_controller import SystemController

    sent = []

    class Bus:
        async def publish(self, message):
            message.to_json_bytes()  # as the pub client sends it
            sent.append(message)

    def init(self, **kw):
        self.comms = SimpleNamespace(create_pub_client=lambda address: Bus())
        self._init_credit_channels(None)

    monkeypatch.setattr(StickyCreditRouter, "__init__", init)
    monkeypatch.setattr(StickyCreditRouter, "_init_credit_channels", lambda s, c: None)
    monkeypatch.setattr(CompassCreditRouter, "exception", lambda self, msg: None)

    async def main():
        rt = StragglerRuntime([(1.0, {0})])
        lp = str(tmp_path / "lp")
        monkeypatch.setattr(TrafficLP, "from_env", classmethod(lambda cls: cls(rt, lp)))
        traffic = CompassCreditRouter(run=None, service_id="timing_manager").traffic
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(addresses(lp)[1])
        traffic.send(_key(0))
        while traffic.now() < 1.0:
            await traffic.wait(1.0)
        push.send_pyobj((_key(0), 1, 0.5, False))
        try:
            await asyncio.Event().wait()  # only the router stops the loop
        finally:
            push.close(linger=0)

    with pytest.raises(RuntimeError, match="Event loop stopped before Future"):
        _run(main())
    error, cancel = sent
    assert isinstance(cancel, ProfileCancelCommand)
    assert cancel.service_id == error.service_id == "timing_manager"

    async def shutdown_check():
        pass

    controller = SimpleNamespace(
        _exit_errors=[],
        _result_join_coordinator=ResultJoinCoordinator(),
        error=lambda msg: None,
        _check_and_trigger_shutdown=shutdown_check,
    )
    asyncio.run(SystemController._process_service_error_message(controller, error))
    (exit_error,) = controller._exit_errors
    assert exit_error.error_details.type == "Straggler"
    assert "seq 1 arrives at 0.5, now 1.0" in exit_error.error_details.message
