# SPDX-License-Identifier: MIT
"""A pipeline stage's send: its completion arithmetic, and the stage loop's
receive, wait and send order around its runner calls.

The stage loop runs on a real `LPRuntime` whose clock authority is the scripted
fake connection of the LP runtime tests; the runner calls are a stand-in that
charges each forward `C` seconds, as the engine's `call_func` does.
"""

import math
from functools import partial
from types import SimpleNamespace

import pytest
import zmq

from atom.compass.clock import LpId, single_engine_table
from atom.compass.runner import pp_send
from atom.compass.runner.overrides import HostDeviceRecurrence
from atom.compass.runner.pp_send import CarrierModel, StageSends
from atom.utils import clock
from atom.utils.clock import LPRuntime, WrappedSocket
from tests.compass.clock.test_lp_runtime import FakeConn, _owner

CARRIER = CarrierModel(1024, 2.0**-10, 2.0**-12, 2.0**-20, 2.0**-8)
MODELS = {"rccl": CARRIER, "gloo": CARRIER}
L_DATA, L_ACK = pp_send.lookaheads(MODELS)
#: The size on gloo, eager, then one tensor on RCCL, rendezvous.
RDV = [("gloo", 8), ("rccl", 2048)]
EAGER = [("gloo", 8)]
#: Both parts rendezvous: the sender's host waits for the ack of the size.
ACKED = [("gloo", 2048), ("rccl", 2048)]
C = 2.0**-6
DATA, ACK = "engine-pp0->engine-pp1:pp_data#dp0", "engine-pp1->engine-pp0:pp_ack#dp0"


def test_a_rendezvous_part_is_posted_when_the_part_before_it_ends():
    sent = 2.0**-12 + 8 * 2.0**-20  # the eager size part, sender side
    first = sent + 2.0**-10
    done, end = pp_send.recv_done(MODELS, 0.0, 0.0, RDV)
    # The tensor leaves at `sent`; posted at `first`, its receive finds the
    # threshold's 1024 bytes, 2**-10 s of them, already moved.
    assert done == end == sent + 2.0**-8 + 2048 * 2.0**-20
    # Posted late, the receive starts the rendezvous part later.
    assert (
        pp_send.recv_done(MODELS, 0.0, 1.0, RDV)[1]
        == 1.0 - 2.0**-10 + 2.0**-8 + 2.0**-9
    )
    # Posted before the send, the rendezvous part starts at the send.
    assert (
        pp_send.recv_done(MODELS, 1.0, 0.0, [("rccl", 2048)])
        == (1.0 + 2.0**-8 + 2.0**-9,) * 2
    )
    assert pp_send.recv_done(MODELS, 0.0, 0.0, EAGER) == (sent, first)
    assert (L_DATA, L_ACK) == (2.0**-10, 2.0**-8 - 2.0**-10)


def test_a_rendezvous_send_moves_its_threshold_before_a_late_receive_once():
    """Two tensors, ready at 0: a receive posted at 1.0 finds the first's
    1024 bytes moved, the second leaving only when the first is done; one
    posted within those bytes' time ends where one posted at the send does."""
    pushed, tensors, T = 1024 * 2.0**-20, [("rccl", 2048)] * 2, 2.0**-8 + 2.0**-9
    assert pp_send.recv_done(MODELS, 0.0, 1.0, tensors) == (1.0 - pushed + 2 * T,) * 2
    assert pp_send.recv_done(MODELS, 0.0, pushed / 2, tensors)[1] == 2 * T


def test_the_ack_lookahead_holds_for_a_receive_posted_after_the_threshold_moved():
    """A gloo carrier eager to 1 MiB: its smallest rendezvous, posted long
    after the send, completes a fixed part and one byte after the post."""
    big = {"gloo": CarrierModel(2**20, 2.0**-10, 0.0, 2.0**-20, 2.0**-8)}
    l_ack = pp_send.lookaheads(big)[1]
    done = pp_send.recv_done(big, 0.0, 4.0, [("gloo", 2**20 + 1)])[0]
    assert done - 4.0 >= l_ack > 0


def test_a_lookahead_that_is_not_positive_is_refused_by_name():
    fast = {"rccl": CarrierModel(1, 2.0**-6, 0.1, 2.0**-30, 2.0**-7)}
    with pytest.raises(
        ValueError, match=r"T_min 0\.0078125\d* is not above L_data 0\.015625"
    ):
        pp_send.lookaheads(fast)
    with pytest.raises(ValueError, match="the gloo carrier's latency_s is 0.0"):
        pp_send.lookaheads({"gloo": CarrierModel(1, 0.0, 0.1, 0.1, 0.1)})


def _stage(rank, *grants, frames=(), parts=ACKED):
    """Stage `rank` of two, its owner thread and the names its runner was called with."""
    table = single_engine_table(
        admission_path="serving",
        ipc_s=0.5,
        stream_s=0.5,
        pp=2,
        pp_links=(L_DATA, L_ACK),
    )
    thread, on_owner = _owner()
    rt = LPRuntime(LpId(f"engine-pp{rank}"), table, FakeConn(*grants), owner=thread)
    rt.start_run()
    frames, calls = list(frames), []

    def call(name, *args, wait_out=False):
        calls.append(name)
        if name == "forward":
            rt.advance_to(rt.now + C)
            return SimpleNamespace(pp_send=(C, 0.0, parts))
        return frames.pop(0) if name == "recv_pp_frame" else True

    sends = StageSends(rt, call, rank, 2, MODELS)
    return rt, partial(on_owner, sends), sends, calls


def _sent(rt):
    """Each clock request's kind, time and the sends it logged."""
    return [m[:3] for m in rt.conn.sent]


def test_an_ack_released_during_the_compute_sends_at_its_end_and_a_later_one_at_done():
    done = C + 2.0**-7  # inside the second forward's compute
    rt, run, _, calls = _stage(
        0,
        (C, {}),
        (2 * C, {ACK: [(0, done)]}),
        (3 * C, {}),
        (4.0, {ACK: [(1, 4.0)]}),
        frames=[
            {"seq": 0, "tensors_done": -math.inf},
            {"seq": 1, "tensors_done": -math.inf},
        ],
    )
    for _ in range(3):
        run("forward")
    assert _sent(rt) == [
        ("TAR", C, []),
        ("TAR", 2 * C, [(DATA, 0, C + L_DATA)]),
        ("TAR", 3 * C, [(DATA, 1, 2 * C + L_DATA)]),
        ("NER", math.inf, []),  # the third send waits for ack 1
    ]
    assert rt.send_log == [(DATA, 2, 4.0 + L_DATA)]
    assert calls.count("recv_pp_frame") == 2


def test_an_idle_flush_ends_at_the_ack_or_the_local_completion_or_where_it_was():
    rt, run, sends, _ = _stage(
        0,
        (C, {}),
        (0.75, {ACK: [(0, 0.75)]}),
        frames=[{"seq": 0, "tensors_done": -math.inf}],
    )
    run("forward")
    run("flush_pp_send")
    assert (rt.now, sends.pending) == (0.75, None)
    run("flush_pp_send")  # nothing pending: no clock call
    assert len(rt.conn.sent) == 2

    rt, run, sends, _ = _stage(
        0, (C, {}), (C + 2.0**-12 + 8 * 2.0**-20, {}), parts=EAGER
    )
    run("forward")
    run("flush_pp_send")
    assert _sent(rt)[1] == ("TAR", C + 2.0**-12 + 8 * 2.0**-20, [(DATA, 0, C + L_DATA)])


def test_a_stage_holds_work_from_its_send_to_the_flush_and_while_kv_work_is_pending(
    monkeypatch,
):
    kv = [False]
    rt, run, stub, _ = _stage(
        0, (C, {}), (C + 2.0**-12 + 8 * 2.0**-20, {}), parts=EAGER
    )
    engine = SimpleNamespace(
        runner_mgr=SimpleNamespace(call_func=stub.call),
        pp_rank=0,
        pp_size=2,
        has_pending_kv_work=lambda: kv[0],
    )
    monkeypatch.setattr(clock, "_installed", rt)
    spec = SimpleNamespace(
        value=lambda key: getattr(CARRIER, key.rsplit(".", 1)[1], 1.0)
    )
    pp_send.install(engine, spec)
    sends = partial(run.func, engine.runner_mgr.call_func)
    held = [rt.holds_work()]
    sends("forward")
    held.append(rt.holds_work())
    sends("flush_pp_send")
    held.append(rt.holds_work())
    kv[0] = True
    assert held + [rt.holds_work()] == [False, True, False, True]


def test_a_stage_waits_for_its_own_ring_and_only_the_last_for_its_device_work():
    decode = SimpleNamespace(
        total_seqs_num_prefill=0, total_tokens_num=1, total_seqs_num=1
    )
    decode.produces_output = lambda: True

    def charges(stage, n=5):
        host = HostDeviceRecurrence({}, (), lambda: 0.0, stage)
        return [host.charge(decode, 1.0, 0.0) for _ in range(n)]

    # Entry waits for the device work `pp` forwards back, not two.
    assert charges((0, 4)) == [0.0, 0.0, 0.0, 0.0, 1.0]
    # The last stage returns once its own device work ends.
    assert charges((3, 4), 1) == [1.0]
    assert charges(None, 3) == [0.0, 1.0, 2.0]


def test_a_bounded_poll_is_essential_while_the_stage_holds_work():
    table = single_engine_table(admission_path="serving", ipc_s=0.5, stream_s=0.5)
    thread, on_owner = _owner()
    rt = LPRuntime(LpId("engine"), table, FakeConn((0.1, {}), (0.2, {})), owner=thread)
    rt.start_run()
    raw = SimpleNamespace(recv_multipart=_nothing_here)
    rx = WrappedSocket(rt, raw, "frontend->engine:request#dp0")
    on_owner(rx.poll, 100)
    rt.holds_work = lambda: True
    on_owner(rx.poll, 100)
    assert [m[1:4:2] for m in rt.conn.sent] == [(math.inf, 0.1), (0.2, 0.2)]


def _nothing_here(*args, **kw):
    raise zmq.Again
