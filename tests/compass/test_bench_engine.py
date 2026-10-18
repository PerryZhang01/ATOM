# SPDX-License-Identifier: MIT
"""The bench starts its runner on every worker through ATOM's runner manager.

The manager is ATOM's own, with its worker processes, ZMQ channels and rank
barrier; two things stand in for what needs a driver. aiter's shared-memory
broadcast, which a driverless import cannot load, is one queue per worker,
and each worker is the stand-in runner below: the window test's planted device
under `BenchWindow` and `StepRecording`, answering the startup calls by
logging them, its workers agreeing over a gloo group.
"""

import itertools
import multiprocessing
import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from test_bench_window import BUSY, MS, PREPARE, Bench, batch

from atom.compass.bench import window

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.compass.bench import engine
    from atom.model_engine import async_proc
    from atom.utils import init_exit_handler

BLOCKS = 7


class Runner(Bench):
    """Rank r takes (1 + 2r) ms to issue a forward, and its model's work lasts
    `BUSY` + r us."""

    def __init__(self, rank, config):
        pc = config.parallel_config
        if config.tp_world_size > 1:
            # Through the store the bench published, as `ModelRunner` does.
            assert pc._managed_distributed_store
            store = dist.TCPStore(
                pc.data_parallel_master_ip, pc.data_parallel_base_port, is_master=False
            )
            dist.init_process_group("gloo", store=store, rank=rank, world_size=2)
        issue, busy = (1 + 2 * rank) * MS, BUSY + 1_000 * rank
        super().__init__(itertools.repeat(issue), itertools.repeat(busy))
        self.calls = []

    def bench_group(self):
        return dist.group.WORLD if dist.is_initialized() else None

    def get_num_blocks(self):
        self.calls.append("get_num_blocks")
        return {"num_kvcache_blocks": BLOCKS}

    def allocate_kv_cache(self, blocks):
        self.calls.append(["allocate_kv_cache", blocks])
        return True

    def capture_cudagraph(self):
        self.calls.append("capture_cudagraph")
        return 0.0, [], 0

    def freeze_gc_heap(self):
        self.calls.append("freeze_gc_heap")
        return 0

    def bench_step(self, batch):
        self.calls.append("bench_step")
        return super().bench_step(batch)

    def call_log(self):
        return self.calls

    def exit(self):
        if dist.is_initialized():
            dist.destroy_process_group()


class _Broadcast:
    """aiter's shared-memory broadcast, as one queue per worker."""

    def __init__(self, workers, *args, **kwargs):
        self.queues = [
            multiprocessing.get_context("spawn").Queue() for _ in range(workers)
        ]

    def export_handle(self):
        return self.queues

    @classmethod
    def create_from_handle(cls, queues, rank):
        reader = cls.__new__(cls)
        reader.queues = [queues[rank]]
        return reader

    def enqueue(self, message):
        for queue in self.queues:
            queue.put(message)

    def dequeue(self):
        return self.queues[0].get()

    def close(self):
        self.closed = True


def _worker(*args, **kwargs):
    """`AsyncIOProc`, in a worker whose aiter is the stub."""
    async_proc.MessageQueue = _Broadcast
    async_proc.AsyncIOProc(*args, **kwargs)


class _Spawn:
    """The spawn context, starting each worker through `_worker`."""

    def __init__(self):
        self.context = multiprocessing.get_context("spawn")

    def Barrier(self, parties):
        return self.context.Barrier(parties)

    def Process(self, target, **kwargs):
        assert target is async_proc.AsyncIOProc
        return self.context.Process(target=_worker, **kwargs)


def _config(tp=1, eager=False, dp=1, dp_attention=False, ep=False, pp=1):
    return SimpleNamespace(
        tp_world_size=tp,
        tensor_parallel_size=tp,
        prefill_context_parallel_size=1,
        parallel_config=SimpleNamespace(
            data_parallel_size=dp,
            data_parallel_rank=0,
            data_parallel_rank_local=0,
            data_parallel_master_ip="127.0.0.1",
            data_parallel_base_port=0,
        ),
        enable_dp_attention=dp_attention,
        enable_expert_parallel=ep,
        pipeline_parallel_size=pp,
        enforce_eager=eager,
        num_kvcache_blocks=None,
    )


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr(engine, "RUNNER", f"{__name__}.Runner")
    monkeypatch.setattr(async_proc, "MessageQueue", _Broadcast)
    monkeypatch.setattr(async_proc, "get_mp_context", _Spawn)
    # The manager would take the test process's SIGINT and SIGTERM.
    monkeypatch.setattr(async_proc, "init_exit_handler", lambda self: None)
    # The bench takes them too.
    handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    yield
    for s, handler in handlers.items():
        signal.signal(s, handler)


@pytest.mark.parametrize("tp, eager", [(2, False), (1, True)])
def test_every_worker_starts_in_engine_cores_order_and_the_slowest_is_charged(
    manager, tp, eager
):
    config = _config(tp=tp, eager=eager)
    bench = engine.Bench(config, BLOCKS)
    try:
        assert config.num_kvcache_blocks == BLOCKS
        result = bench.measure(batch())
        log = bench.runner_mgr.call_func("call_log", wait_out=True)
    finally:
        bench.exit()
    assert bench.runner_mgr.rpc_broadcast_mq.closed

    startup = ["get_num_blocks", ["allocate_kv_cache", BLOCKS]]
    capture = [] if eager else ["capture_cudagraph"]
    assert log == startup + capture + ["freeze_gc_heap", "bench_step"]
    # Rank 0's reply carries rank 1's times, and its preload outlasted rank 1's
    # host: at 3 ms rank 1's entry event had completed, and rank 0's had not.
    slowest = tp - 1
    busy = BUSY + 1_000 * slowest
    assert result["repeats"] == [[PREPARE + busy, PREPARE, busy]] * window.REPEATS
    assert result["preload_ns"] == 2 * (1 + 2 * slowest) * MS
    assert result["discarded"] == 1


def test_the_bench_holds_the_pool_it_is_given_not_the_one_its_budget_gives(manager):
    config = _config(eager=True)
    bench = engine.Bench(config, BLOCKS + 2)
    try:
        log = bench.runner_mgr.call_func("call_log", wait_out=True)
    finally:
        bench.exit()
    pool = ["allocate_kv_cache", BLOCKS + 2]
    assert log == ["get_num_blocks", pool, "freeze_gc_heap"]
    assert config.num_kvcache_blocks == BLOCKS + 2


class Small(Runner):
    """A device that holds the `BLOCKS` its budget gives, and no more."""

    def allocate_kv_cache(self, blocks):
        if blocks > BLOCKS:
            raise torch.OutOfMemoryError("HIP out of memory")
        return super().allocate_kv_cache(blocks)


def test_a_device_that_cannot_hold_the_pool_stops_the_bench_naming_both_counts(
    manager, monkeypatch
):
    monkeypatch.setattr(engine, "RUNNER", f"{__name__}.Small")
    refusal = (
        f"a pool of {BLOCKS + 2} KV blocks, and this device's budget gives {BLOCKS}$"
    )
    with pytest.raises(ValueError, match=refusal):
        engine.Bench(_config(eager=True), BLOCKS + 2)


class Dies(Runner):
    """A worker that raises in the call its config's `dies_in` names."""

    def __init__(self, rank, config):
        super().__init__(rank, config)
        setattr(self, config.dies_in, self.die)

    def die(self, *args):
        raise RuntimeError("HIP out of memory")


@pytest.mark.parametrize(
    "call", ["get_num_blocks", "capture_cudagraph", "freeze_gc_heap", "bench_step"]
)
def test_a_worker_that_dies_in_a_call_stops_the_bench_naming_the_call(
    manager, monkeypatch, call
):
    """Not the manager's bare `SystemExit`, which ends a command with status 0
    and no message."""
    monkeypatch.setattr(engine, "RUNNER", f"{__name__}.Dies")
    config = _config()
    config.dies_in = call
    with pytest.raises(ValueError, match=f"a bench worker died, so {call} got no"):
        bench = engine.Bench(config, BLOCKS)
        try:
            bench.measure(batch())
        finally:
            bench.exit()


class Signals(Dies):
    """A worker that, in the call its config's `dies_in` names, sends SIGTERM to
    the process that started the bench twice, the second while the bench tears
    down, as GNU `timeout`'s second delivery can, then outlasts the teardown."""

    def die(self, *args):
        for _ in range(2):
            os.kill(os.getppid(), signal.SIGTERM)
            time.sleep(0.5)
        time.sleep(60)


@pytest.mark.parametrize("call", ["allocate_kv_cache", "bench_step"])
def test_a_sigterm_stops_the_bench_naming_it_and_a_repeat_leaves_the_teardown_whole(
    manager, monkeypatch, call
):
    """With the manager's own SIGTERM handler installed, which stops the
    workers as a dead worker does."""
    monkeypatch.setattr(engine, "RUNNER", f"{__name__}.Signals")
    monkeypatch.setattr(async_proc, "init_exit_handler", init_exit_handler)
    monkeypatch.setenv("ATOM_SHUTDOWN_TIMEOUT_S", "2")
    config = _config(eager=True)
    config.dies_in = call
    with pytest.raises(engine.Signalled, match="its process was sent SIGTERM$"):
        bench = engine.Bench(config, BLOCKS)
        try:
            bench.measure(batch())
        finally:
            bench.exit()
    assert not multiprocessing.active_children()


def test_a_stop_from_the_keyboard_reaches_the_process_that_started_the_bench(
    manager, monkeypatch
):
    """The manager ignores SIGINT where it starts, as ATOM's own handler does."""
    ignore = lambda self: signal.signal(signal.SIGINT, signal.SIG_IGN)
    monkeypatch.setattr(async_proc, "init_exit_handler", ignore)
    previous = signal.getsignal(signal.SIGINT)
    try:
        bench = engine.Bench(_config(eager=True), BLOCKS)
        bench.exit()
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    finally:
        signal.signal(signal.SIGINT, previous)


@pytest.mark.parametrize(
    "refused, what",
    [
        ({"dp": 2}, "2 data-parallel ranks"),
        ({"tp": 2, "dp_attention": True}, "2 data-parallel ranks"),
        ({"ep": True}, "expert parallelism"),
        ({"pp": 2}, "pipeline parallelism"),
    ],
)
def test_a_parallelism_the_bench_does_not_build_refuses_before_any_worker_starts(
    monkeypatch, refused, what
):
    monkeypatch.setattr(engine, "AsyncIOProcManager", None)  # never called
    with pytest.raises(ValueError, match=f"does not measure under {what}"):
        engine.Bench(_config(**refused), BLOCKS)
