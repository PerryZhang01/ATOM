# SPDX-License-Identifier: MIT
"""A DP member parks before its forward when the deployment offloads to `lmcache_mp`.

A member waiting on the lmcache MP server asks for time alone while its peers,
their forwards dispatched, wait on it in the predicted forward's gather. Each
peer's park stands in its rounds, the member's time moves alone, and the last
park releases every member at its clock. Between the DP state sync and the
park a member asks no TAR.
"""

import json
import math
import queue
import threading
import uuid
from types import SimpleNamespace

import pytest
import zmq
from test_dp_member_step_end import collectives  # noqa: F401
from test_dp_slice import _deployment
from test_vertical_slice import _run_file

from atom.compass import clock_transport
from atom.compass import run as compass_run
from atom.compass.clock import NER, PARK, TAR, ClockAbort, ClockAuthority, LpId
from atom.model_engine.async_proc import AsyncIOProcManager
from atom.model_engine.block_table_codec import BlockTableDeltaEncoder
from atom.model_engine.engine_core import DPEngineCoreProc
from atom.utils import clock
from atom.utils.clock import LPRuntime, WrappedSocket
from atom.utils.distributed import utils

INF = math.inf
ENGINE, FRONTEND, TRAFFIC = compass_run.ENGINE, compass_run.FRONTEND, LpId("traffic")
REQUEST = "frontend->engine:request#dp1"
LMCACHE_MP = {"kv_connector": "lmcache_mp", "kv_role": "offload"}
STEP_S = 0.002


def _authority(tmp_path):
    run = json.loads(_run_file(tmp_path, data_parallel_size=2).read_text())
    ca = compass_run._RecordingAuthority(run)
    ca.on_request(TRAFFIC, NER, INF, [])
    return ca


def _grants(replies):
    return [(to[1], g) for to, g, _ in replies]


def test_peers_park_while_one_member_waits_and_all_leave_at_its_now(tmp_path):
    ca = _authority(tmp_path)
    assert ca.on_request(FRONTEND, NER, INF, [(REQUEST, 0, 0.25)]) == []
    assert ca.on_request(ENGINE, PARK, INF, [], member="dp1") == []
    # dp0 alone: dp1's park stands in each round, and its request is held.
    for t, g in ((0.3, 0.25), (0.3, 0.3), (0.5, 0.5)):
        assert _grants(ca.on_request(ENGINE, NER, t, [], member="dp0")) == [("dp0", g)]

    replies = ca.on_request(ENGINE, PARK, INF, [], member="dp0")

    assert _grants(replies) == [("dp0", 0.5), ("dp1", 0.5)]
    released = {to[1]: r for to, _, r in replies}
    assert released["dp1"][REQUEST] == [(0, 0.25)]
    assert not any(released["dp0"].values())
    # From the clock the grant that released it found, not the release's.
    assert [str(r) for r in ca.steps.rows if r.event == "release"] == [
        f"engine 0.0 0.25 release {REQUEST} 0 -"
    ]


def test_a_tar_meeting_a_parked_member_is_refused(tmp_path):
    ca = _authority(tmp_path)
    ca.on_request(ENGINE, PARK, INF, [], member="dp1")
    with pytest.raises(ClockAbort, match="dp0 asked TAR while dp1 asked PARK"):
        ca.on_request(ENGINE, TAR, 1.0, [], member="dp0")


def test_a_parked_member_leaves_at_the_finish(tmp_path):
    ca = _authority(tmp_path)
    ca.on_request(FRONTEND, NER, INF, [])
    ca.on_request(ENGINE, PARK, INF, [], member="dp1")
    replies = ca.on_request(ENGINE, NER, INF, [], member="dp0")
    assert ((ENGINE, "dp1"), INF) in [(to, g) for to, g, *_ in replies]


class _IdleRank(DPEngineCoreProc):
    """A DP rank past its construction with nothing to schedule and an offload
    connector: ATOM's own state sync, step, idle drain and dummy run on it."""

    def __init__(self, schedule, reply):
        self._shutting_down, self.dp_group = False, None
        self.kv_transfer_enabled, self._next_idle_kv_drain = True, 0.0
        self.polled = []
        self.scheduler = SimpleNamespace(
            schedule=schedule,
            take_rejected=list,
            publish_kv_events=lambda: None,
            kv_connector=SimpleNamespace(
                is_offload=True, build_connector_meta=lambda: None
            ),
            _update_from_kv_xfer_finished=self.polled.append,
        )
        mgr = object.__new__(AsyncIOProcManager)
        mgr.label, mgr.proc_num, mgr.kv_outputs_queues = "dp0", 1, [queue.Queue()]
        mgr.kv_output_aggregator = mgr._pending_kv_aggregation = None
        mgr._block_table_encoder = BlockTableDeltaEncoder()
        mgr.rpc_broadcast_mq = SimpleNamespace(enqueue=lambda msg: None)
        mgr.outputs_queue = queue.Queue()
        mgr.outputs_queue.put(reply)
        self.runner_mgr = mgr


@pytest.fixture
def dp2(monkeypatch, tmp_path, collectives):  # noqa: F811
    """A DP2 run file; what entering an engine changes is put back after."""
    path = _run_file(tmp_path, data_parallel_size=2)
    monkeypatch.setenv(compass_run.ENV, str(path))
    monkeypatch.setattr(compass_run, "_member", None)
    monkeypatch.setattr(clock, "_installed", None)
    monkeypatch.setattr(utils, "LP_OF_RANK", None, raising=False)
    for cls, name in (
        (AsyncIOProcManager, "call_func"),
        (DPEngineCoreProc, "_sync_dp_state"),
    ):
        monkeypatch.setattr(cls, name, getattr(cls, name))
    return json.loads(path.read_text())


def _enter(rank, kv=LMCACHE_MP):
    deployment = _deployment(dp=2, rank=rank)
    deployment.kv_transfer_config = kv
    with compass_run.engine(deployment):
        pass
    return clock.installed()


@pytest.mark.parametrize(
    "kv, parks",
    [
        ({}, False),
        ({"kv_connector": "mooncake"}, False),
        (
            {
                "kv_connector": "multi",
                "connectors": [{"kv_connector": "mooncake"}, LMCACHE_MP],
            },
            True,
        ),
    ],
)
def test_only_a_member_that_offloads_to_lmcache_mp_parks(monkeypatch, dp2, kv, parks):
    call = AsyncIOProcManager.call_func
    monkeypatch.setattr(compass_run, "connect", lambda *_: None)
    _enter(1, kv)
    assert (AsyncIOProcManager.call_func is not call) == parks


def test_the_check_refuses_a_tar_after_the_dp_state_sync(monkeypatch, dp2):
    monkeypatch.setattr(compass_run, "connect", lambda *_: None)
    rt = _enter(0)
    _IdleRank(None, None)._sync_dp_state(False, False, False)
    with pytest.raises(RuntimeError, match="between the DP state sync and the forward"):
        rt.advance_to(1.0)
    assert rt.refusals and rt.calls == 0


def _thread(target):
    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t


def test_an_idle_step_that_waits_alone_parks_and_releases_its_peer_at_its_now(dp2):
    endpoint, table = dp2["clock_endpoint"], compass_run.channel_table(dp2)
    server = clock_transport.serve(
        ClockAuthority(table, members=compass_run._members(table, 2)), endpoint
    )
    traffic = clock_transport.connect(TRAFFIC, endpoint)
    traffic.send((NER, INF, [], INF))
    ctx, address = zmq.Context(), f"inproc://dp-park-{uuid.uuid4().hex}"
    push, pull = ctx.socket(zmq.PUSH), ctx.socket(zmq.PULL)
    push.bind(address)
    pull.connect(address)
    left, handled = [], []

    def frontend():
        rt = LPRuntime(FRONTEND, table, clock_transport.connect(FRONTEND, endpoint))
        rt.start_run()
        rt.advance_to(0.25)
        WrappedSocket(rt, push, REQUEST).send(b"r")
        rt.next_event(INF)

    def dp1():  # dispatched at once; its step loop parks first
        peer.start_run()
        peer.park()
        left.append(peer.now)
        peer.advance_group([peer.now + STEP_S] * 2, peer.now + STEP_S)
        peer.next_event(INF)

    def receive():  # dp1's input thread
        sock = WrappedSocket(peer, pull, REQUEST)
        handled.append((sock.recv(), peer.now))
        peer.back_at_wait_point()

    member = threading.Thread(target=dp1, daemon=True)
    peer = LPRuntime(
        ENGINE, table, clock_transport.connect(ENGINE, endpoint, "dp1"), owner=member
    )
    member.start()
    threads = [_thread(receive), _thread(frontend), member]
    rt = _enter(0)

    def schedule():  # a lookup that polls at 0.3, then waits until 0.5
        for t in (0.3, 0.5):
            while rt.now < t:
                rt.next_event(t)

    rank = _IdleRank(
        schedule, SimpleNamespace(predicted_s=STEP_S, group_targets=(STEP_S,) * 2)
    )
    rank._sync_dp_state(False, False, False)
    assert rank._process_engine_step() is False
    rank._execute_dummy_batch()

    assert rt.now == 0.5 + STEP_S and rank.polled == [None] and not rt.refusals
    rt.next_event(INF)
    for t in threads:
        t.join(timeout=10)
    assert left == [0.5]
    assert handled == [(b"r", 0.25 + table.lookahead(REQUEST))]
    server.close()
    ctx.destroy(linger=0)
