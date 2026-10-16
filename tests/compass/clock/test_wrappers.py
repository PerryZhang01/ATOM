# SPDX-License-Identifier: MIT
"""`WrappedSocket`, `WrappedPoller` and `RelayQueue` over real zmq ``inproc`` pairs.

Each test builds the socket pair ATOM uses on that channel: ROUTER to DEALER for
requests, PUSH to PULL for the engine's output. The clock authority is the
scripted fake connection of the LP runtime tests, and each LP's clock owner is
a real thread, so a release that never completes fails the test instead of
hanging it.
"""

import math
import re
import threading
import time
from types import SimpleNamespace

import pytest
import zmq

from atom.compass.clock import LpId, prefill_decode_table
from atom.utils.clock import (
    LPRuntime,
    RelayQueue,
    UnsentRelayItem,
    WrappedPoller,
    WrappedSocket,
)

from .test_lp_runtime import ENGINE, FRONTEND, IPC, OUT, REQ, FakeConn, _owner, _table


@pytest.fixture
def ctx():
    ctx = zmq.Context()
    yield ctx
    ctx.destroy(linger=0)


def _pair(ctx, name, send_type, recv_type):
    tx, rx = ctx.socket(send_type), ctx.socket(recv_type)
    rx.bind(f"inproc://{name}")
    tx.connect(f"inproc://{name}")
    return tx, rx


def _rt(lp, *replies, table=None):
    thread, call = _owner()
    conn = FakeConn(*replies)
    rt = LPRuntime(lp, table or _table(), conn, owner=thread)
    rt.start_run()
    return rt, call


def _requests(rt):
    """Each clock request's kind, time and send log; later fields are not checked."""
    return [m[:3] for m in rt.conn.sent]


def _relayed(rt, raw, ch, *stamps):
    """A sending wrapper whose stamps are scripted, as a relay would hand them over."""
    ws = WrappedSocket(rt, raw, ch)
    ws.relay = SimpleNamespace(take_stamp=iter(stamps).__next__)
    return ws


def test_out_of_order_frames_are_handed_over_in_arrival_order_and_counted_by_seq(ctx):
    """One channel, seq 1 physically delivered before seq 0."""
    push, pull = _pair(ctx, "out", zmq.PUSH, zmq.PULL)
    tx = _relayed(_rt(ENGINE)[0], push, OUT, (10.1, 1), (10.05, 0))
    rt, call = _rt(FRONTEND, (10.2, {OUT: [(0, 10.05), (1, 10.1)]}))
    rx = WrappedSocket(rt, pull, OUT)
    stop_tx, stop_rx = _pair(ctx, "stop", zmq.PAIR, zmq.PAIR)
    seen = []

    def output_thread():
        poller = WrappedPoller(rt)
        poller.register(stop_rx, zmq.POLLIN)
        poller.register(rx, zmq.POLLIN)
        while True:
            socks = poller.poll()
            if socks[0][0] is stop_rx:
                return
            seen.append((rx.recv(copy=False).bytes, rt.read_clock()))

    thread = threading.Thread(target=output_thread, daemon=True)
    thread.start()
    tx.send(b"p1")
    tx.send(b"p0")
    call(rt.advance_to, 10.2)
    assert seen == [(b"p0", 10.05), (b"p1", 10.1)]
    assert rt.handled[OUT] == {0, 1}
    assert rt.now == 10.2
    stop_tx.send(b"")
    thread.join(5)
    assert not thread.is_alive()


def test_the_relay_sends_every_put_stamped_at_its_put(ctx):
    push, pull = _pair(ctx, "out", zmq.PUSH, zmq.PULL)
    rt, call = _rt(ENGINE, (0.5, {}), (0.7, {}))
    tx = WrappedSocket(rt, push, OUT)
    relay = RelayQueue(rt, tx)
    rt.end_run()
    relay.put(b"ready")  # outside the run: any thread, nothing stamped
    tx.send(relay.get())
    rt.start_run()
    call(relay.put, b"a")
    call(rt.advance_to, 0.5)
    call(relay.put_nowait, b"b")
    call(rt.advance_to, 0.7)
    call(relay.put, b"c")
    for _ in range(3):
        tx.send(relay.get())
    with pytest.raises(RuntimeError, match=f"^{OUT}: one relay item sent twice"):
        tx.send(b"c")

    rx_rt, _ = _rt(FRONTEND)
    rx = WrappedSocket(rx_rt, pull, OUT)
    while len(rx.buf) < 4:
        assert pull.poll(5000)
        rx.pull()
    assert [f[3].bytes for f in rx.buf] == [b"ready", b"a", b"b", b"c"]
    stamps = {(OUT, 0): IPC, (OUT, 1): 0.5 + IPC, (OUT, 2): 0.7 + IPC}
    assert rx_rt.unreleased == stamps
    assert _requests(rt) == [
        ("TAR", 0.5, [(OUT, 0, IPC)]),
        ("TAR", 0.7, [(OUT, 1, 0.5 + IPC)]),
    ]

    call(relay.put, b"d")
    call(relay.put, b"e")
    relay.get()  # taken, never sent
    unsent = f"{OUT}: the item stamped (arrival {0.7 + IPC}, seq 3) was taken"
    with pytest.raises(UnsentRelayItem, match="^" + re.escape(unsent)):
        relay.get()
    with pytest.raises(RuntimeError, match="^stamp_send from thread 'MainThread'"):
        relay.put(b"f")

    refused = []
    second = threading.Thread(target=lambda: refused.append(_raised(relay.get)))
    second.start()
    second.join(5)
    assert "the relay's one consumer is 'MainThread'" in refused[0]


def _raised(fn):
    try:
        fn()
    except RuntimeError as e:
        return str(e)


def test_a_router_puts_the_header_after_the_identity_frame(ctx):
    router, dealer = _pair(ctx, "req", zmq.ROUTER, zmq.DEALER)
    dealer.send(b"")
    identity, _ = router.recv_multipart()
    tx_rt, call = _rt(FRONTEND)
    tx = WrappedSocket(tx_rt, router, REQ)
    rt, _ = _rt(ENGINE)
    rx = WrappedSocket(rt, dealer, REQ)

    tx_rt.end_run()
    call(lambda: tx.send_multipart([identity, b"ready"], copy=False))
    assert dealer.poll(5000)
    rx.pull()
    assert rx.ready() and rt.arrived[REQ] == set()
    assert rx.recv() == b"ready"

    tx_rt.start_run()
    call(lambda: tx.send_multipart([identity, b"req"], copy=False))
    assert dealer.poll(5000)
    rx.pull()
    assert rt.unreleased == {(REQ, 0): IPC}
    assert [f[3].bytes for f in rx.buf] == [b"req"]


def test_a_duplicate_seq_is_refused_on_read(ctx):
    push, pull = _pair(ctx, "out", zmq.PUSH, zmq.PULL)
    tx = _relayed(_rt(ENGINE)[0], push, OUT, (1.0, 3), (1.0, 3))
    rx = WrappedSocket(_rt(FRONTEND)[0], pull, OUT)
    tx.send(b"x")
    tx.send(b"x")
    with pytest.raises(RuntimeError, match=f"^{OUT} seq 3 arrived twice"):
        while len(rx.buf) < 2:
            assert pull.poll(5000)
            rx.pull()


KV = "engine-D->engine-P:kv_write_req"


def _pd_table():
    return prefill_decode_table(
        admission_path="serving",
        ipc_s=IPC,
        stream_s=0.002,
        router_s=0.001,
        kv_write_req_s=0.5,
    )


def test_inline_receive_idles_until_the_frame_is_released(ctx):
    table = _pd_table()
    push, pull = _pair(ctx, "kv", zmq.PUSH, zmq.PULL)
    tx = _relayed(_rt(LpId("engine-D"), table=table)[0], push, KV, (0.6, 1), (0.5, 0))
    tx.send(b"p1")
    tx.send(b"p0")
    grant = (0.6, {KV: [(0, 0.5), (1, 0.6)]})
    rt, call = _rt(LpId("engine-P"), (0.1, {}), (0.3, {}), grant, table=table)
    rx = WrappedSocket(rt, pull, KV)

    while len(rx.buf) < 2:
        assert pull.poll(5000)
        rx.pull()
    assert call(rx.poll, 0) is False
    assert rt.conn.sent == []
    assert call(rx.poll, 100) is False
    assert rt.conn.sent == [("NER", math.inf, [], 0.1)]
    assert call(rx.recv) == b"p0"
    assert _requests(rt)[1:] == [("NER", math.inf, [])] * 2
    assert call(rx.recv) == b"p1"
    assert len(rt.conn.sent) == 3
    assert rt.handled[KV] == {0, 1} and rt.now == 0.6


def test_settle_waits_for_a_released_frame_still_in_flight(ctx):
    table = _pd_table()
    push, pull = _pair(ctx, "kv", zmq.PUSH, zmq.PULL)
    tx = _relayed(_rt(LpId("engine-D"), table=table)[0], push, KV, (0.5, 0))
    rt, call = _rt(LpId("engine-P"), (0.5, {KV: [(0, 0.5)]}), table=table)
    rx = WrappedSocket(rt, pull, KV)
    call(rt.next_event, math.inf)  # released before it is sent
    settling = threading.Thread(target=rx.settle, daemon=True)
    settling.start()
    settling.join(0.2)
    assert settling.is_alive()
    tx.send(b"p0")
    settling.join(5)
    assert not settling.is_alive() and rt.arrived[KV] == {0}


def test_inline_receive_takes_a_frame_released_while_still_in_flight(ctx):
    """Released by the receive's own grant, or by an earlier `advance_to`, then sent."""
    table = _pd_table()
    push, pull = _pair(ctx, "kv", zmq.PUSH, zmq.PULL)
    stamps = (0.5, 0), (0.55, 1), (0.7, 2)
    tx = _relayed(_rt(LpId("engine-D"), table=table)[0], push, KV, *stamps)
    grants = [(a, {KV: [(seq, a)]}) for a, seq in stamps]
    rt, call = _rt(LpId("engine-P"), *grants, table=table)
    rx = WrappedSocket(rt, pull, KV)

    def send_later(payload):
        threading.Timer(0.2, lambda: tx.send(payload)).start()

    send_later(b"p0")
    assert call(rx.recv) == b"p0"
    send_later(b"p1")
    assert call(rx.poll, 100) is True
    assert call(rx.recv) == b"p1"
    call(rt.advance_to, 0.7)
    send_later(b"p2")
    assert call(rx.recv) == b"p2"
    assert _requests(rt) == [("NER", math.inf, [])] * 2 + [("TAR", 0.7, [])]
    assert rt.conn.sent[1][3] == 0.6


def test_outside_the_run_an_inline_receive_waits_on_the_socket(ctx):
    """Before the run and after the +inf grant, as ATOM's loops poll until shutdown."""
    table = _pd_table()
    push, pull = _pair(ctx, "kv", zmq.PUSH, zmq.PULL)
    tx_rt, _ = _rt(LpId("engine-D"), table=table)
    tx_rt.end_run()
    tx = WrappedSocket(tx_rt, push, KV)
    rt, call = _rt(LpId("engine-P"), (math.inf, {}), table=table)
    rt.end_run()
    rx = WrappedSocket(rt, pull, KV)

    def outside_the_run(asked):
        start = time.monotonic()
        assert call(rx.poll, 100) is False
        assert _requests(rt) == asked
        assert time.monotonic() - start >= 0.05
        threading.Timer(0.2, lambda: tx.send(b"x")).start()
        assert call(rx.recv) == b"x"
        assert _requests(rt) == asked

    outside_the_run([])
    rt.start_run()
    call(rt.next_event, math.inf)
    rt.end_run()  # the window closes at the +inf grant
    outside_the_run([("NER", math.inf, [])])
