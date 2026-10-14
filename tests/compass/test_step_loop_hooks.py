# SPDX-License-Identifier: MIT
"""The engine step loops on an LP clock: TAR by what a reply charges, NER when idle.

ATOM's own loops (`EngineCore.busy_loop`, `DPEngineCoreProc.busy_loop`,
`PPEngineCoreProc._head_busy_loop`) and its worker RPC
(`AsyncIOProcManager.call_func`) run on an engine whose scheduler, workers and
utility handler are fakes. The clock authority is a fake for one LP that knows
the arrival schedule: a grant delivers each request whose arrival it reaches
into the engine's input queue, as the input thread would.
"""

import queue
import random
import threading
from itertools import accumulate
from types import SimpleNamespace

import pytest
from aiter_stub import stubbed_aiter

from atom.compass.clock import LpId, single_engine_table
from atom.model_engine.block_table_codec import BlockTableDeltaEncoder
from atom.model_engine.scheduler import ScheduledBatchOutput
from atom.model_engine.sequence import SequenceStatus
from atom.utils import clock
from atom.utils.clock import LPRuntime
from tests.compass.clock.test_lp_runtime import _owner

with stubbed_aiter():
    from atom.kv_transfer.disaggregation import KVConnectorOutput
    from atom.model_engine import async_proc
    from atom.model_engine.async_proc import AsyncIOProcManager
    from atom.model_engine.engine_core import (
        KV_IDLE_DRAIN_INTERVAL_S,
        DPEngineCoreProc,
        EngineCore,
    )
    from atom.model_engine.engine_core_mgr import CoreManager
    from atom.model_engine.llm_engine import LLMEngine
    from atom.model_engine.pp_engine_core import PPEngineCoreProc

INF = float("inf")
TABLE = single_engine_table(admission_path="serving", ipc_s=0.001, stream_s=0.002)
STEP_S = 0.01
STEPS = 3  # steps per request
#: 64 requests, Poisson at 8 per second: the schedule the arrival barrier spun on.
_rng = random.Random(0)
ARRIVALS = list(accumulate(_rng.expovariate(8.0) for _ in range(64)))


class Authority:
    """Grants a TAR its target; an NER the earliest of its targets and the next arrival.

    With no arrival left and no essential target, it grants ``+inf``: daemon
    deadlines do not keep the run alive.
    """

    def __init__(self, arrivals, deliver, busy=lambda: False):
        self.arrivals, self.deliver, self.busy = list(arrivals), deliver, busy
        self.calls = []  # (kind, the clock when asked, target, grant)
        self.busy_ners = 0  # NERs asked with a request still to run

    def send(self, msg):
        self.msg = msg

    def recv(self):
        kind, t, _, t_daemon = self.msg
        self.busy_ners += kind == "NER" and self.busy()
        if kind == "TAR":
            G = t
        elif self.arrivals or t < INF:
            G = min(t, t_daemon, *self.arrivals[:1])
        else:
            G = INF
        while self.arrivals and self.arrivals[0] <= G:
            self.deliver(self.arrivals.pop(0))
        self.calls.append((kind, self.rt.now, t, G))
        return G, {}


class Workers:
    """`AsyncIOProcManager` without processes: a step answers `reply`, others ``None``."""

    call_func = AsyncIOProcManager.call_func
    label = "workers"

    def __init__(self, reply):
        self.outputs_queue, self.steps = queue.Queue(), 0
        self._block_table_encoder = BlockTableDeltaEncoder()
        self.rpc_broadcast_mq = SimpleNamespace(enqueue=self.enqueue)
        self.reply = reply

    def enqueue(self, msg):
        step = msg[0] in ("forward", "dummy_execution")
        self.steps += step
        self.outputs_queue.put(self.reply if step else None)


class Scheduler:
    """Runs every waiting request in each step, `STEPS` steps per request."""

    prefill_delayer = None
    metrics = SimpleNamespace(record_forward=lambda batch, seqs: None)

    def __init__(self):
        self.waiting = []

    def is_finished(self):
        return not self.waiting

    def extend(self, seqs):
        self.waiting += seqs

    def schedule(self):
        if not self.waiting:
            return None
        seqs = {s.id: s for s in self.waiting}
        return SimpleNamespace(req_ids=list(seqs), connector_meta_output=None), seqs

    def postprocess(self, seqs, fwd_out, **kw):
        for s in seqs:
            s.left -= 1
        self.waiting = [s for s in self.waiting if s.left]
        return [s for s in seqs if not s.left]

    def heartbeat_throughput(self, now):
        pass

    def take_rejected(self):
        return []

    def compute_detailed_aggregates(self, *args):
        pass

    def publish_kv_events(self):
        pass

    def shutdown_kv_events(self):
        pass


class Engine(EngineCore):
    """ATOM's engine methods over fake workers, scheduler and utility handler."""

    _execute_dummy_batch = DPEngineCoreProc._execute_dummy_batch
    _pp_head_step = EngineCore._process_engine_step

    def __init__(self, reply):
        self.input_queue, self.output_queue = queue.Queue(), queue.Queue()
        self.stream_output_queue, self.utility_queue = queue.Queue(), queue.Queue()
        self.scheduler, self.runner_mgr = Scheduler(), Workers(reply)
        self.pushes = []
        self.utility_handler = SimpleNamespace(
            process_queue=lambda q, engine: None,
            push_metrics=lambda: self.pushes.append(clock.installed().now),
        )
        self.label, self.kv_transfer_enabled = "engine", False
        self._is_rl_weights_offloaded, self._next_idle_kv_drain = False, 0.0
        self._weight_sync_fenced = False
        self.engines_running, self._in_flight = True, []

    def _sync_dp_state(self, unfinished, shutdown, offloaded):
        return unfinished, shutdown, offloaded


@pytest.fixture
def installed():
    yield
    clock.install(None)


def _install(ca, lp="engine", start=True):
    """Install an LP runtime over `ca`; returns `call`, which runs a function on its owner."""
    thread, call = _owner()
    ca.rt = LPRuntime(LpId(lp), TABLE, ca, owner=thread)
    clock.install(ca.rt)
    if start:
        ca.rt.start_run()
    return call


def _run(loop, reply, arrivals=ARRIVALS):
    """Run `loop` on an engine LP over the fake authority; returns (engine, authority)."""
    engine = Engine(reply)
    ca = Authority(
        arrivals,
        lambda a: engine.input_queue.put(
            [SimpleNamespace(id=a, status=SequenceStatus.WAITING, left=STEPS)]
        ),
        busy=lambda: not engine.scheduler.is_finished(),
    )
    _install(ca)(loop, engine)
    return engine, ca


#: Each loop, and the times it pushes metrics: every 5 LP seconds from 0, the
#: interval the test sets.
LOOPS = {
    "EngineCore": (EngineCore.busy_loop, [0.0, 5.0]),
    "DPEngineCoreProc": (DPEngineCoreProc.busy_loop, [0.0, 5.0]),
    "PPEngineCoreProc head": (PPEngineCoreProc._head_busy_loop, [0.0, 5.0]),
}


@pytest.mark.parametrize("name", LOOPS)
def test_an_idle_loop_asks_for_time_once_per_idle_period(name, installed, monkeypatch):
    monkeypatch.setenv("ATOM_METRICS_UPDATE_INTERVAL_S", "5")
    loop, pushes = LOOPS[name]
    engine, ca = _run(loop, SimpleNamespace(predicted_s=STEP_S))
    served = []
    while not engine.output_queue.empty():
        served += engine.output_queue.get()
    ners = [G for kind, _, _, G in ca.calls if kind == "NER"]
    tars = [(now, t) for kind, now, t, _ in ca.calls if kind == "TAR"]
    at_arrival = len(set(ners) & set(ARRIVALS))
    print(
        f"{name}: {len(ners)} NER ({at_arrival} at an arrival, "
        f"{len(set(ners) & set(engine.pushes))} at a metrics push, 1 at +inf) and "
        f"{len(tars)} TAR for {len(ARRIVALS)} requests"
    )
    assert sorted(s.id for s in served) == ARRIVALS
    assert ners[-1] == INF and ca.rt.now == INF
    # Each NER is granted at an arrival, a metrics push or the finish, and never
    # twice at one time: an idle period costs one request, not a spin.
    assert set(ners) <= set(ARRIVALS) | set(engine.pushes) | {INF}
    assert len(set(ners)) == len(ners) and ca.busy_ners == 0
    assert engine.pushes == pushes
    assert len(tars) == engine.runner_mgr.steps
    assert all(t == pytest.approx(now + STEP_S) for now, t in tars)


def test_a_reply_advances_the_clock_only_by_the_seconds_it_carries(installed):
    ca = Authority([], None)
    call = _install(ca)
    workers = Workers(SimpleNamespace())
    call(lambda: workers.call_func("forward", wait_out=True))
    assert ca.calls == []
    workers.reply = ScheduledBatchOutput([], [], None, None, None, predicted_s=0.5)
    call(lambda: workers.call_func("dummy_execution", wait_out=True))
    assert ca.calls == [("TAR", 0.0, 0.5, 0.5)]


def test_with_pending_kv_work_an_idle_point_waits_one_drain_interval(installed):
    ca = Authority([], None)
    call = _install(ca, start=False)
    engine = Engine(None)
    engine.has_pending_kv_work = lambda: True
    assert call(clock.idle, engine._idle_deadline, 5.0) is False
    assert ca.calls == []  # before the run there is no clock to ask
    ca.rt.start_run()
    for _ in range(2):  # an idle point that follows an idle point waits again
        call(clock.idle, engine._idle_deadline, 5.0)
    D = KV_IDLE_DRAIN_INTERVAL_S
    assert ca.calls == [("NER", 0.0, D, D), ("NER", D, 2 * D, 2 * D)]


@pytest.mark.parametrize("simulated", [True, False])
def test_a_kv_poll_waits_for_its_reply_only_on_a_simulated_run(
    installed, simulated, monkeypatch
):
    """A poll that finds a reply missing returns None, and the scheduler then skips
    that poll's `process_completions`: a simulated transfer would be reported at
    a step set by how fast a worker answered in wall time."""
    if simulated:
        clock.install(LPRuntime(LpId("engine"), TABLE, conn=None))

    def answer(msg):  # 50 wall ms after each request, as a worker process would
        reply = mgr.kv_outputs_queues[0].put
        threading.Timer(0.05, reply, (KVConnectorOutput(),)).start()

    # The manager's own constructor, with no worker process, socket or signal.
    process = SimpleNamespace(start=lambda: None)
    ctx = SimpleNamespace(Barrier=lambda n: None, Process=lambda **kw: process)
    mq = SimpleNamespace(export_handle=lambda: None, enqueue=answer)
    for name, value in {
        "get_mp_context": lambda: ctx,
        "MessageQueue": lambda *args, **kwargs: mq,
        "init_exit_handler": lambda self: None,
    }.items():
        monkeypatch.setattr(async_proc, name, value)
    for name in ("process_output_sockets", "process_kv_output_sockets"):
        monkeypatch.setattr(AsyncIOProcManager, name, lambda *args: None)
    monkeypatch.setattr(AsyncIOProcManager, "monitor_procs", lambda self: None)
    mgr = AsyncIOProcManager(None, 1, "workers.Runner")
    got = mgr.call_func_with_aggregation("async_proc_aggregation")
    assert (got is not None) == simulated


def test_with_no_runtime_the_hooks_read_nothing():
    class Untouchable:
        def __getattr__(self, name):
            raise AssertionError(f"read {name}")

    reply = Untouchable()
    assert clock.idle(Untouchable(), 0.0) is False
    assert clock.charge(reply) is reply
    clock.wait_output(Untouchable())
    clock.finish()
    clock.close()


def test_get_output_idles_until_an_output_and_refuses_at_the_finish(installed):
    mgr = SimpleNamespace(outputs_queue=queue.Queue())
    ca = Authority([1.0, 2.0], lambda a: a == 2.0 and mgr.outputs_queue.put(["done"]))
    call = _install(ca, "frontend")
    assert call(CoreManager.get_output, mgr) == ["done"]
    assert [(k, G) for k, _, _, G in ca.calls] == [("NER", 1.0), ("NER", 2.0)]
    with pytest.raises(RuntimeError, match="no output left"):
        call(CoreManager.get_output, mgr)


def test_the_frontend_stops_the_engines_only_after_the_finish(installed):
    ca = Authority([1.0], lambda a: None)
    call = _install(ca, "frontend")
    in_run_when_stopped = []
    engine = SimpleNamespace(
        core_mgr=SimpleNamespace(close=lambda: in_run_when_stopped.append(ca.rt.in_run))
    )
    call(LLMEngine.close, engine)
    assert [(k, G) for k, _, _, G in ca.calls] == [("NER", 1.0), ("NER", INF)]
    assert in_run_when_stopped == [False]  # the engines stop after the finish
    call(LLMEngine.close, engine)  # after the finish there is nothing to ask
    assert len(ca.calls) == 2


@pytest.mark.parametrize("loop_raises", [False, True])
def test_an_engine_leaving_before_the_finish_is_refused_unless_it_raised(
    monkeypatch, installed, loop_raises
):
    """A loop that returns at 3.0 is refused; a loop that raised keeps its own error."""

    def busy_loop(self):
        if loop_raises:
            raise ValueError("the loop's own error")

    monkeypatch.setattr(EngineCore, "_setup_engine_process", lambda name: None)
    monkeypatch.setattr(EngineCore, "__init__", lambda self, *args: None)
    monkeypatch.setattr(EngineCore, "busy_loop", busy_loop)
    monkeypatch.setattr(EngineCore, "exit", lambda self: None)
    clock.install(LPRuntime(LpId("engine"), TABLE, conn=None))
    clock.installed().now = 3.0
    config = SimpleNamespace(
        pipeline_parallel_size=1, parallel_config=SimpleNamespace(data_parallel_size=1)
    )
    error, match = (
        (ValueError, "own error") if loop_raises else (RuntimeError, "left its loop")
    )
    with pytest.raises(error, match=match):
        EngineCore.run_engine(config, "in", "out")
