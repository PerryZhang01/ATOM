# SPDX-License-Identifier: MIT
"""A DP group member returns from a group step at its own step end.

Each rank's target comes from the predicted forward's gather. Two members of
the engine LP, joined by the clock authority, take a step whose targets are
80.5 ms (dp1, an idle rank's dummy) and 244.7 ms (dp0). dp1 is the test's own
thread, so its intake is read between its return and the collective that
settles it; dp0 and the frontend run on threads of their own.
"""

import math
import queue
import threading
import uuid
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import zmq
from test_dp_slice import _deployment
from test_dp_step_max import dp2, prefill, runner  # noqa: F401
from test_vertical_slice import _run_file

from atom.compass import clock_transport
from atom.compass import run as compass_run
from atom.compass.audit import sync_scan
from atom.compass.backends.step_stub import Coefficients, StepStubBackend
from atom.compass.clock import NER, ClockAuthority, LpId, single_engine_table
from atom.compass.runner.overrides import HostDeviceRecurrence
from atom.utils import clock
from atom.utils.clock import LPRuntime, WrappedSocket
from atom.utils.distributed import utils

TABLE = single_engine_table(
    admission_path="serving", ipc_s=2.0**-14, stream_s=2.0**-12, dp=2
)
ENGINE, FRONTEND = LpId("engine"), LpId("frontend")
REQUEST = "frontend->engine:request#dp1"
TARGETS = (0.0805, 0.2447)
SENDS = (0.05, 0.15)  # one arrives before dp1's step end, one inside its window


@pytest.fixture
def collectives(monkeypatch):
    """Each collective a no-op recording the caller's LP clock, put back after."""
    seen = []
    for name in clock.COLLECTIVES:
        monkeypatch.setattr(
            dist, name, lambda *a, **k: seen.append(clock.installed().now)
        )
    return seen


class Conn:
    """A clock connection granting each request its own target."""

    def __init__(self):
        self.sent = []

    def send(self, request):
        self.sent.append(request[:2])

    def recv(self):
        return self.sent[-1][1], {}


def _thread(target, *args):
    t = threading.Thread(target=target, args=args, daemon=True)
    t.start()
    return t


def test_the_early_member_takes_in_only_what_arrived_by_its_own_step_end(
    monkeypatch, collectives
):
    endpoint = f"inproc:test-dp-member-{uuid.uuid4().hex}"
    server = clock_transport.serve(
        ClockAuthority(TABLE, members=compass_run._members(TABLE, 2)), endpoint
    )
    clock_transport.connect(LpId("traffic"), endpoint).send(
        (NER, math.inf, [], math.inf)
    )
    ctx, address = zmq.Context(), f"inproc://dp-request-{uuid.uuid4().hex}"
    push = ctx.socket(zmq.PUSH)
    push.bind(address)
    pull = ctx.socket(zmq.PULL)
    pull.connect(address)

    def frontend():
        rt = LPRuntime(FRONTEND, TABLE, clock_transport.connect(FRONTEND, endpoint))
        rt.start_run()
        sock = WrappedSocket(rt, push, REQUEST)
        for t in SENDS:
            rt.advance_to(t)
            sock.send(str(t).encode())
        rt.next_event(math.inf)

    def dp0():
        rt = LPRuntime(ENGINE, TABLE, clock_transport.connect(ENGINE, endpoint, "dp0"))
        rt.start_run()
        rt.advance_group(TARGETS, TARGETS[1])
        returned.append(rt.now)
        rt.next_event(math.inf)

    rt = LPRuntime(ENGINE, TABLE, clock_transport.connect(ENGINE, endpoint, "dp1"))
    rt.start_run()
    monkeypatch.setattr(clock, "_installed", rt)
    clock.settle_before_collectives()
    intake, returned = queue.Queue(), []
    sock = WrappedSocket(rt, pull, REQUEST)

    def receive():  # the engine's input thread: one recv per send, then its wait point
        for _ in SENDS:
            intake.put(sock.recv())
        rt.back_at_wait_point()

    threads = [_thread(receive), _thread(frontend), _thread(dp0)]

    rt.advance_group(TARGETS, TARGETS[0])

    assert rt.now == TARGETS[0]
    assert [intake.get_nowait()] == [b"0.05"] and intake.empty()
    with pytest.raises(RuntimeError, match="before the owed group round"):
        rt.advance_to(0.1)
    with pytest.raises(RuntimeError, match="owing group rounds"):
        rt.next_event(math.inf)
    assert returned == []

    dist.all_reduce(None)

    assert collectives == [TARGETS[1]]
    assert intake.get(timeout=10) == b"0.15"
    assert rt.owed == [] and len(rt.refusals) == 2
    rt.next_event(math.inf)
    for t in threads:
        t.join(timeout=10)
    assert returned == [TARGETS[1]]
    server.close()
    ctx.destroy(linger=0)


def test_a_tar_past_the_owed_rounds_joins_them_first():
    rt = LPRuntime(ENGINE, TABLE, Conn())

    rt.advance_group(TARGETS, TARGETS[0])
    rt.advance_to(0.3)

    assert rt.conn.sent == [("TAR", TARGETS[0]), ("TAR", TARGETS[1]), ("TAR", 0.3)]


def test_after_settling_a_pass_that_ran_nothing_waits_at_its_idle_point(monkeypatch):
    # As both members' step loops: a pass after the step, which settles in its
    # collective and runs no step, then a pass that must wait for its next event.
    rt = LPRuntime(ENGINE, TABLE, Conn())
    rt.start_run()
    monkeypatch.setattr(clock, "_installed", rt)
    rt.advance_group(TARGETS, TARGETS[0])
    clock.idle(lambda: math.inf)
    rt.settle()

    clock.idle(lambda: 1.0)

    assert rt.conn.sent[-1] == (NER, 1.0)


def test_an_idle_ranks_dummy_returns_at_its_own_seconds_a_busy_rank_at_the_max(
    dp2,  # noqa: F811
):
    # Prefill dearer than decode, so the busy rank sets the group's max.
    coefficients = Coefficients.constant(prefill_seconds=0.005, decode_seconds=0.001)
    ranks = [
        runner(StepStubBackend(coefficients), dp_size=2, dp_rank=r) for r in (0, 1)
    ]

    busy, idle = dp2(lambda: ranks[0].forward(prefill(40)), ranks[1].dummy_execution)

    assert [busy.predicted_s, idle.predicted_s] == [0.005, 0.001]
    assert busy.group_targets == idle.group_targets == (0.005, 0.001)


def test_every_collective_the_inventory_lists_on_torch_distributed_settles_first(
    monkeypatch, tmp_path, collectives
):
    monkeypatch.setenv(compass_run.ENV, str(_run_file(tmp_path, data_parallel_size=2)))
    monkeypatch.setattr(compass_run, "connect", lambda *_: None)
    monkeypatch.setattr(compass_run, "_member", None)
    monkeypatch.setattr(clock, "_installed", None)
    monkeypatch.setattr(utils, "LP_OF_RANK", None, raising=False)
    with compass_run.engine(_deployment(dp=2, rank=1)):
        pass
    settled = []
    stub = SimpleNamespace(
        owner=threading.current_thread(), now=0.0, settle=lambda: settled.append(1)
    )
    monkeypatch.setattr(clock, "_installed", stub)
    calls = {
        s["call"].rpartition(".")[2]
        for s in sync_scan.load_inventory()["sites"]
        if s["shape"] in ("collective", "barrier")
        and s["call"].rpartition(".")[0] in ("torch.distributed", "dist")
    }
    assert {"all_reduce", "barrier", "all_gather"} <= calls

    for name in sorted(calls):
        getattr(dist, name)()

    assert len(settled) == len(calls)
    _thread(dist.barrier).join()
    assert len(settled) == len(calls)


@pytest.mark.parametrize("device, settles", [("cpu", True), ("meta", False)])
def test_a_host_collective_settles_first_and_a_device_collective_does_not(
    monkeypatch, collectives, device, settles
):
    settled = []
    stub = SimpleNamespace(
        owner=threading.current_thread(), now=0.0, settle=lambda: settled.append(1)
    )
    monkeypatch.setattr(clock, "_installed", stub)
    clock.settle_before_collectives()
    t = torch.zeros(2, device=device)

    dist.all_reduce(t)
    dist.all_gather([t, t], torch.zeros(2))
    dist.broadcast(tensor=t, src=0)

    assert len(settled) == (3 if settles else 0)


@pytest.mark.parametrize("second", ["prefill", "dummy"])
def test_a_member_returns_at_its_own_release_after_the_groups_device_end(
    dp2,  # noqa: F811
    second,
):
    # dp0's prefill is launch-bound, 50 ms of launch over 5 ms of device; dp1's
    # forward has no host time. The forwards' collectives hold dp1's device
    # until dp0's launch ends, but dp1 returns at its own release, before dp0.
    coefficients = Coefficients.constant(prefill_seconds=0.005, decode_seconds=0.001)
    ranks = [
        runner(StepStubBackend(coefficients), dp_size=2, dp_rank=r) for r in (0, 1)
    ]
    launch = {"prefill": {"W_launch": [0.05, 0, 0]}}
    ranks[0].compass_host = HostDeviceRecurrence(
        launch, ("E", "W_pre", "W_post"), lambda: 0.0
    )
    ranks[1].compass_host = HostDeviceRecurrence({}, (), lambda: 0.0)
    step = {"prefill": lambda: ranks[1].forward(prefill(8))}
    step["dummy"] = ranks[1].dummy_execution

    busy, other = dp2(lambda: ranks[0].forward(prefill(40)), step[second])

    assert [busy.predicted_s, other.predicted_s] == [0.05, 0.0]
    assert busy.group_targets == other.group_targets == (0.05, 0.0)
    hosts = [r.compass_host for r in ranks]
    assert [h.ends[-1] for h in hosts] == [0.05, 0.05]
    assert hosts[1].output_end == 0.05


def test_a_member_returns_at_its_own_pull_with_what_arrived_before_it(
    monkeypatch, collectives
):
    # dp1 pulls 30 ms after its release, dp0 10 ms after its own; a request
    # sent at 0.1 s is inside dp1's window from its release to its pull.
    pulls = (TARGETS[0] + 0.03, TARGETS[1] + 0.01)
    sends = (0.05, 0.1)
    endpoint = f"inproc:test-dp-pull-{uuid.uuid4().hex}"
    server = clock_transport.serve(
        ClockAuthority(TABLE, members=compass_run._members(TABLE, 2)), endpoint
    )
    clock_transport.connect(LpId("traffic"), endpoint).send(
        (NER, math.inf, [], math.inf)
    )
    ctx, address = zmq.Context(), f"inproc://dp-pull-{uuid.uuid4().hex}"
    push, pull = ctx.socket(zmq.PUSH), ctx.socket(zmq.PULL)
    push.bind(address)
    pull.connect(address)
    returned, intake = [], queue.Queue()

    def frontend():
        rt = LPRuntime(FRONTEND, TABLE, clock_transport.connect(FRONTEND, endpoint))
        rt.start_run()
        sock = WrappedSocket(rt, push, REQUEST)
        for t in sends:
            rt.advance_to(t)
            sock.send(str(t).encode())
        rt.next_event(math.inf)

    def dp0():  # asks the same rounds as `clock.charge` on dp1
        rt = LPRuntime(ENGINE, TABLE, clock_transport.connect(ENGINE, endpoint, "dp0"))
        rt.start_run()
        rt.advance_group((*TARGETS, *pulls), pulls[1])
        returned.append(rt.now)
        rt.next_event(math.inf)

    rt = LPRuntime(ENGINE, TABLE, clock_transport.connect(ENGINE, endpoint, "dp1"))
    rt.start_run()
    monkeypatch.setattr(clock, "_installed", rt)
    clock.settle_before_collectives()
    sock = WrappedSocket(rt, pull, REQUEST)

    def receive():
        for _ in sends:
            intake.put(sock.recv())
        rt.back_at_wait_point()

    threads = [_thread(receive), _thread(frontend), _thread(dp0)]
    reply = SimpleNamespace(
        predicted_s=TARGETS[0], out_s=0.03, group_targets=TARGETS, group_out=pulls
    )

    clock.charge(reply)

    assert rt.now == pulls[0]
    assert [intake.get(timeout=10), intake.get(timeout=10)] == [b"0.05", b"0.1"]
    dist.all_reduce(None)
    assert collectives == [pulls[1]]
    rt.next_event(math.inf)
    for t in threads:
        t.join(timeout=10)
    assert returned == [pulls[1]]
    server.close()
    ctx.destroy(linger=0)


def test_each_rank_gathers_its_pull_beside_its_release(dp2):  # noqa: F811
    ranks = [runner(StepStubBackend(), dp_size=2, dp_rank=r) for r in (0, 1)]
    for r, (charge_s, out_s) in zip(ranks, [(0.05, 0.002), (0.001, None)]):
        r.compass_host = SimpleNamespace(charge=lambda *_, s=charge_s: s, out_s=out_s)

    busy, idle = dp2(lambda: ranks[0].forward(prefill(40)), ranks[1].dummy_execution)

    assert busy.group_targets == idle.group_targets == (0.05, 0.001)
    assert busy.group_out == idle.group_out == (0.05 + 0.002,)
    assert [busy.out_s, idle.out_s] == [0.002, None]
