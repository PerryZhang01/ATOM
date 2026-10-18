# SPDX-License-Identifier: MIT
"""The bench: `EngineCore` without its scheduler.

`Bench(config, num_blocks)` publishes the workers' rendezvous store as
`CoreManager` does for an engine, then starts `RUNNER` on every worker through
ATOM's runner manager, in `EngineCore`'s order: `get_num_blocks`, which sizes
the per-request state, `allocate_kv_cache` with `num_blocks`, which then
becomes `config.num_kvcache_blocks`, `capture_cudagraph` unless the engine
runs eagerly, and `freeze_gc_heap`. The pool is the one the batches ran on,
not the count `get_num_blocks` reports: that count reads the device's memory
outside torch, which moves from one start to the next, so a pool sized by it
can be smaller than the one a batch filled. A worker that dies stops the
bench, naming the call that got no reply; a device that cannot hold the pool
stops it naming both counts.
`measure(batch)` runs `bench_step` on every worker and returns rank 0's reply,
the window's result (`window.py`). Data parallelism above one, expert
parallelism and pipeline parallelism refuse.

The manager leaves SIGINT ignored in the process that starts it, as an engine
core's process wants; the bench's is the command's, so the bench restores
Python's handler there, and a stop from the keyboard stops the bench, while
the workers keep ignoring it.
"""

import signal

from torch.distributed import TCPStore

from atom.model_engine.async_proc import AsyncIOProcManager

RUNNER = "atom.compass.bench.runner.BenchModelRunner"


class Bench:
    def __init__(self, config, num_blocks: int) -> None:
        # Under DP attention every TP rank becomes a DP rank.
        dp = config.parallel_config.data_parallel_size * (
            config.tensor_parallel_size if config.enable_dp_attention else 1
        )
        for refused, what in (
            (dp > 1, f"{dp} data-parallel ranks"),
            (config.enable_expert_parallel, "expert parallelism"),
            (config.pipeline_parallel_size > 1, "pipeline parallelism"),
        ):
            if refused:
                raise ValueError(f"the bench does not measure under {what}")
        pc = config.parallel_config
        self.store = TCPStore(
            pc.data_parallel_master_ip,
            pc.data_parallel_base_port,
            is_master=True,
            wait_for_workers=False,
        )
        pc.data_parallel_base_port = self.store.port
        pc._managed_distributed_store = True
        self.still_running = True
        self.runner_mgr = AsyncIOProcManager(
            self.exit,
            config.tp_world_size * config.prefill_context_parallel_size,
            RUNNER,
            config,
        )
        signal.signal(signal.SIGINT, signal.default_int_handler)
        try:
            budget = self._call("get_num_blocks")
            try:
                allocated = self.runner_mgr.call_func(
                    "allocate_kv_cache", num_blocks, wait_out=True
                )
            except SystemExit as stop:
                # The manager stops, with a bare SystemExit, when a worker dies.
                raise ValueError(
                    f"the bench stopped allocating a pool of {num_blocks} KV "
                    "blocks, and this device's budget gives "
                    f"{budget['num_kvcache_blocks']}"
                ) from stop
            assert allocated, "Failed to allocate kv cache"
            config.num_kvcache_blocks = num_blocks
            if not config.enforce_eager:
                self._call("capture_cudagraph")
            # Unfrozen, a worker's collections walk its startup heap, and their
            # pauses would discard repeats.
            self._call("freeze_gc_heap")
        except BaseException:
            self.exit()
            raise

    def measure(self, batch) -> dict:
        """`batch`'s step, measured on every worker."""
        return self._call("bench_step", batch)

    def _call(self, name: str, *args):
        """Rank 0's reply to `name`, called on every worker."""
        try:
            return self.runner_mgr.call_func(name, *args, wait_out=True)
        except SystemExit as stop:
            # The manager stops, with a bare SystemExit, when a worker dies.
            raise ValueError(f"a bench worker died, so {name} got no reply") from stop

    def exit(self) -> None:
        """Stop every worker; the manager calls it too, when a worker dies."""
        if not self.still_running:
            return
        self.still_running = False
        if self.runner_mgr.still_running:
            self.runner_mgr.keep_monitoring = False
            self.runner_mgr.call_func("exit")
            self.runner_mgr.exit()
        # Left to the garbage collector at interpreter exit, the broadcast
        # queue's ZMQ context never terminates and the process never exits.
        self.runner_mgr.rpc_broadcast_mq.close()
