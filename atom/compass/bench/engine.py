# SPDX-License-Identifier: MIT
"""The bench: `EngineCore` without its scheduler.

`Bench(config)` publishes the workers' rendezvous store as `CoreManager` does
for an engine, then starts `RUNNER` on every worker through ATOM's runner
manager, in `EngineCore`'s order: `get_num_blocks`, `allocate_kv_cache` with the blocks
it reports, which then become `config.num_kvcache_blocks`, and
`capture_cudagraph` unless the engine runs eagerly. `measure(batch)` runs
`bench_step` on every worker and returns rank 0's reply, the window's result
(`window.py`). Data parallelism above one, expert parallelism and pipeline
parallelism refuse.
"""

from torch.distributed import TCPStore

from atom.model_engine.async_proc import AsyncIOProcManager

RUNNER = "atom.compass.bench.runner.BenchModelRunner"


class Bench:
    def __init__(self, config) -> None:
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
        try:
            blocks = self.runner_mgr.call_func("get_num_blocks", wait_out=True)
            blocks = blocks["num_kvcache_blocks"]
            assert self.runner_mgr.call_func(
                "allocate_kv_cache", blocks, wait_out=True
            ), "Failed to allocate kv cache"
            config.num_kvcache_blocks = blocks
            if not config.enforce_eager:
                self.runner_mgr.call_func("capture_cudagraph", wait_out=True)
        except BaseException:
            self.exit()
            raise

    def measure(self, batch) -> dict:
        """`batch`'s step, measured on every worker."""
        return self.runner_mgr.call_func("bench_step", batch, wait_out=True)

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
