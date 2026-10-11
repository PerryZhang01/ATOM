# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The model-runner methods that keep construction off the device.

These are the memory-owning methods of ATOM's `ModelRunner`: the ones that read
a checkpoint, warm the model, size the KV pool, allocate it, capture graphs, and
run a step. They live here rather than beside the subclass because this module
imports nothing from the engine at module scope, and so can be executed on a
machine with no driver.

Two of them run during construction (`_build_and_load_model`, `_maybe_warmup`)
and decline to do their work. The rest run afterwards, over the worker's RPC
channel; `allocate_kv_cache` does the arithmetic and none of the allocation,
`capture_cudagraph` captures nothing and says so in the shape its caller
unpacks, `forward` reports what a step produced without running one, and
`get_num_blocks` runs ATOM's own budget arithmetic against readings taken off a
machine spec instead of off a card.

That last one replaces no arithmetic, and the two methods beside it are the
reason it does not have to. `ModelRunner.get_num_blocks` touches the device in
exactly two calls -- `_read_device_memory` and `_estimate_cudagraph_overhead`
-- and both are override points, so replacing them leaves every line of the
budget formula, the clamp and `plan_pools` to run unchanged. A second copy of
that formula would drift from it with no test failing; a substituted reading
cannot.

Order matters when mixing this in: `NonAllocatingRunner` must precede
`ModelRunner` in the bases so these definitions win. The class deliberately
defines no `__init__`. The base class runs the whole of its own `__init__`
before a subclass body would get control, so anything these methods read has to
come from `self.config`, which the base sets first.

## The reply contract, and why it is not a matter of taste

A worker runs `AsyncIOProc.busy_loop`, which takes a name off a shared-memory
ring, resolves it with `getattr(runner, name, None)`, calls it, and forwards the
result **only when it is not None** (`atom/model_engine/async_proc.py`). The
caller's side is `AsyncIOProcManager.call_func`, whose `wait_out=True` form
blocks on `self.outputs_queue.get()` with no timeout. Three consequences, none
of which a single-process test can show:

* **A name the runner does not have is skipped, not raised.** `getattr` returns
  None, the loop moves on, the worker stays healthy, and a caller that asked for
  the reply waits for the life of the process. Absence is the quietest failure
  on this surface, which is why `RPC_SURFACE` below names every dispatched
  method rather than leaving it to whatever the class happens to inherit.
* **A method that is present and answers None parks its caller in exactly the
  same way.** This is one line below the skip, not a separate mechanism: in
  `busy_loop`, **both** of the loop's `put_nowait` calls -- the primary output
  queue and the KV queue -- are inside `if out is not None:`, just after the
  `getattr` skip. So from the caller's side an answer of None and a method that
  was never defined are one event: nothing is queued, nothing logs, and no
  timeout ends either wait.
  Naming only absence is worse than saying nothing, because it sends whoever
  is debugging the hang to check whether the method is there, find that it is,
  and stop. The shapes here are therefore checked against what the call site
  unpacks, not against what reads well -- returning None is the case a
  plausible stub falls into by accident, which is why every name on the
  surface is either replaced here or left to a base implementation that is
  known to end in a value.
* **Raising is the loud option, and it is louder in the worker than in the
  parent.** An exception leaves `busy_loop`, kills the worker, and the
  manager's process monitor turns that into a `SystemExit` on the output queue,
  which `call_func` re-raises in the caller. What crosses the boundary is the
  type and nothing else: the parent gets a bare `SystemExit()` with empty args,
  no message and no `__cause__`, so a refusal's name and its reason exist only
  on the worker's stderr. `SystemExit()` also carries `code=None`, and the
  `get_num_blocks` call in `EngineCore.__init__`
  (`atom/model_engine/engine_core.py`) sits in a `try/finally` with no
  `except`, so letting it propagate ends the parent process with status 0 -- a
  clean shutdown to any supervisor reading exit codes. So a refusal reaches the
  caller and a silence
  never does, but "reaches" means the fact of it and not the reason for it: a
  successor that wants its refusal diagnosable in the engine's own log has to
  put it there itself, on the worker side, before it raises.

The worker control commands are refused the third way: by a reply. A raise
would end the engine for a command a client sent while serving, and a None
would park it, so `RefusedControlCommands` answers each with its reason,
`command:<method>`. `busy_loop` forwards it, `EngineUtilityHandler` logs it
and, for all but `update_weights`, sends it back to the client as the
`UTILITY_RESPONSE` result; the runner also keeps every reason it answered for
the run summary's refusals.

`RPC_SURFACE` records, for each dispatched name, whether its caller waits for
the reply. The names themselves are not a list anyone typed: they are the ones
`engine_core`, `pp_engine_core` and `engine_utility` broadcast that a
`ModelRunner` answers, and the tests derive that intersection from ATOM's own
source and compare it with this table.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from atom.compass.backends.base import CostBackend
from atom.compass.memory import EAGER_SOURCE, DeviceReadings, SizedKVPool
from atom.compass.runner.step_output import (
    DeferredTokenStream,
    filler_token_ids,
    reports_previous_step,
)

logger = logging.getLogger(__name__)

# Every name broadcast to a worker whose runner is a `ModelRunner`, and whether
# the caller blocks on the reply. True means a None return, or a missing method,
# parks that caller; False means a reply is not read and lands on the output
# queue for whoever asks next. A call site is named beside each one -- but what
# a site fixes varies, and the difference matters: where the comment says the
# caller unpacks, subscripts or reads the reply, the site fixes the shape; where
# it says the reply is forwarded or dropped, the site fixes only that something
# non-None and picklable has to arrive, and any richer shape is a convention
# inherited from the base implementation rather than a requirement.
#
# Scope: a runner reached through `EngineCore`. Nothing else in the tree says
# so, and it is load-bearing. Other dispatched names are absent from this
# table because `ModelRunner` does not define them -- some belong to
# `RapidServeModelRunner`, the rest to the rollout extension. The rollout names
# do reach this runner: `AsyncLLMEngine.__init__` (`atom/rollout/async_engine.py`)
# installs its own runner with `kwargs.setdefault`, so a caller that names this
# one gets it, and `RefusedControlCommands` answers each with a refusal.
# The RapidServe names are kept out by `Config`, not by
# this module: `enable_rapidserve` picks `PrefillEngineCore` /
# `DecodeEngineCore` (`LLMEngine.__init__`), those classes broadcast all of
# them with `wait_out=True` without consulting `runner_qualname`, and the
# substitution that installs a RapidServe runner (`Config.__post_init__`) fires
# only while `runner_qualname` is still ATOM's default -- which Compass
# overwrites. `Config` therefore raises `ValueError` for
# `enable_rapidserve=True` with any runner not in `RAPIDSERVE_RUNNERS`, this
# one included.
#
# Call sites below are in `atom/model_engine/`.
RPC_SURFACE: dict[str, bool] = {
    # Replaced here, in this module.
    "get_num_blocks": True,  # EngineCore.__init__ reads four keys off a dict
    "allocate_kv_cache": True,  # EngineCore.__init__ asserts the reply
    "capture_cudagraph": True,  # EngineCore.__init__ unpacks three values
    "forward": True,  # Scheduler.postprocess reads its attributes
    # Answered by ATOM's own, which needs neither weights nor a device for them.
    "dummy_execution": True,  # DPEngineCoreProc._execute_dummy_batch returns it
    "exit": False,  # EngineCore.exit, the last call of the process's life
    "freeze_gc_heap": True,  # EngineCore._freeze_after_startup catches a raise, not a None
    "process_kvconnector_output": False,  # EngineCore does not wait
    "async_proc_aggregation": True,  # the one bounded wait, in EngineCore
    "flush_pp_send": True,  # PPEngineCoreProc waits before the next send
    # Refused by `RefusedControlCommands`: ATOM's own starts a real profiler.
    "start_profiler": True,  # EngineUtilityHandler forwards the reply unread
    "stop_profiler": True,  # EngineUtilityHandler forwards the reply unread
}


def unanswered_rpc_names(runner: Any) -> tuple[str, ...]:
    """The dispatched names `getattr` would answer with None for this runner.

    Drawn from all twelve, which do not fail the same way, so this return
    carries no one story about its names. `RPC_SURFACE[name]` records whether
    any caller reads the reply, and this function's only caller in the package
    partitions the result on exactly that before reporting it: a waited name
    parks its caller on an unbounded queue read for the life of the process,
    while `exit` and `process_kvconnector_output` are read by nobody and a
    hole in either parks no one.

    Checked where the class is composed instead of being discovered by a
    deployment, because the worker raises on none of them.
    """
    return tuple(name for name in RPC_SURFACE if getattr(runner, name, None) is None)


class RunnerRefusal(RuntimeError):
    """Raised where this runner has no answer and will not invent one."""


def install_device_readings(runner: Any, readings: DeviceReadings) -> None:
    """Give a runner the readings its KV budget will be computed from.

    Set on the instance rather than taken in an `__init__`, because the base
    class runs the whole of its own before a subclass body would get control.
    It is a function rather than a method so that every method on
    `NonAllocatingRunner` is a replacement for one of ATOM's, which is a
    property the tests check rather than a preference.

    Reading a machine spec off a flag or a config field, and building the
    readings from it, is the configuration surface's job and is not wired yet.
    Until it is, whoever has the spec installs the readings here, and
    `get_num_blocks` refuses by name in the meantime rather than sizing a pool
    from nothing.
    """
    if not isinstance(readings, DeviceReadings):
        raise RunnerRefusal(
            "the KV budget is sized from the five substituted readings, and "
            f"{type(readings).__name__} is not them; install what "
            "`atom.compass.memory.device_readings` returned for this width"
        )
    runner.compass_readings = readings


def _installed_readings(runner: Any) -> DeviceReadings:
    """The installed readings, or a refusal that says what would supply them."""
    readings = getattr(runner, "compass_readings", None)
    if readings is None:
        raise RunnerRefusal(
            "no device readings are installed on this runner, so there is "
            "nothing for ATOM's KV budget arithmetic to run against; call "
            "`install_device_readings` with what "
            "`atom.compass.memory.device_readings` produced from a machine "
            "spec before the engine asks this worker for a block count."
        )
    return readings


def install_cost_backend(runner: Any, backend: CostBackend) -> None:
    """Give a runner the cost backend its steps are priced by.

    Set on the instance the way `install_device_readings` sets the readings,
    and for the same reason. Choosing a backend from a flag or a config field
    is the configuration surface's job and is not wired yet.
    """
    if not isinstance(backend, CostBackend):
        raise RunnerRefusal(
            "a step is priced by a `CostBackend`, and "
            f"{type(backend).__name__} is not one"
        )
    runner.compass_backend = backend


def _installed_backend(runner: Any) -> CostBackend:
    """The installed backend, or a refusal that says what would supply it.

    A step with no backend would take no time, which reads as a valid run.
    """
    backend = getattr(runner, "compass_backend", None)
    if backend is None:
        raise RunnerRefusal(
            "no cost backend is installed on this runner, so there is nothing "
            "to price a step with; call `install_cost_backend` before the "
            "engine's first step."
        )
    return backend


class HostStream:
    """Stands in for a device stream on a runner that starts on the host.

    It has no methods, so code that tries to use it as a stream fails on the
    attribute it asked for instead of queueing work on a device.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


def start_on_host(runner: Any, rank: int, config: Any) -> None:
    """Start a TP1 runner on the host: no device set, its groups on gloo.

    Replaces ATOM's `_setup_device_and_distributed`, which
    `ModelRunner.__init__` calls before it builds any buffer, so every buffer
    the rest of `__init__` builds on `runner.device` lands on the host. Two touches there do not follow the
    device: `torch.cuda.Stream` refuses `cpu`, and `CpuGpuBuffer` pins its host
    half, which is a driver allocation. Both are replaced for the life of the
    worker process, which hosts this runner and nothing else.

    The groups are the ones ATOM's own call builds, on gloo. At one rank
    aiter builds no device communicator. At data-parallel width the DP and EP
    groups span the DP ranks, so they are built with an empty custom group
    configuration, under which aiter builds no device communicator for any
    group. A start wider within one engine is refused: a TP, PCP or PP group
    would need one, and a simulated TP width needs `apply_simulated_tp` on a
    device group.
    """
    pc = config.parallel_config
    ranks = (
        config.tensor_parallel_size
        * config.prefill_context_parallel_size
        * config.pipeline_parallel_size
    )
    if ranks != 1:
        raise RunnerRefusal(
            f"this start is {ranks} ranks wide within its engine; a runner "
            "starts on the host only at one rank per engine, because aiter "
            "builds a TP, PCP or PP group's device communicators on the device."
        )
    from aiter.dist.utils import get_distributed_init_method

    from atom.utils import CpuGpuBuffer

    pinned = CpuGpuBuffer.__init__

    def unpinned(buffer: Any, *size: Any, pin_memory: bool = True, **kw: Any):
        pinned(buffer, *size, pin_memory=False, **kw)

    CpuGpuBuffer.__init__ = unpinned
    torch.cuda.Stream = HostStream
    runner.device = torch.device("cpu")
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = str(config.port)
    method = get_distributed_init_method(
        pc.data_parallel_master_ip, pc.data_parallel_base_port
    )
    if pc.data_parallel_size == 1:
        from aiter import init_dist_env

        init_dist_env(
            1, rankID=rank, backend="gloo", distributed_init_method=method, local_rank=0
        )
        return
    from aiter.dist.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=1,
        rank=rank,
        distributed_init_method=method,
        backend="gloo",
        local_rank=0,
        data_parallel_size=pc.data_parallel_size,
        data_parallel_rank=pc.data_parallel_rank,
    )
    initialize_model_parallel(
        1, 1, data_parallel_size=pc.data_parallel_size, custom_group_config={}
    )


def _config_field(runner: Any, name: str) -> Any:
    """A config field read with no default; a missing one is refused by name."""
    try:
        return getattr(runner.config, name)
    except AttributeError:
        raise RunnerRefusal(
            f"ATOM's config has no field {name!r}; this runner reads it with "
            "no default, so a config that lacks it is refused by name rather "
            "than read as if the field were unset."
        ) from None


class UnbuiltModel(torch.nn.Module):
    """Stands in for the module tree a non-allocating runner never builds.

    It registers no parameter and no buffer, so it costs nothing on any device,
    and calling it raises instead of returning something a caller could mistake
    for a forward pass.
    """

    def __init__(self, model_class: Any) -> None:
        super().__init__()
        self.model_class_name = getattr(model_class, "__name__", repr(model_class))

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise RunnerRefusal(
            f"{self.model_class_name} was never built: this runner predicts what a "
            "forward pass costs instead of running one, so there are no weights "
            "to call."
        )


def _refused(method: str):
    """A worker method that answers with its refusal instead of doing its work."""

    def refuse(self, *args: Any) -> str:
        reason = f"command:{method}"
        vars(self).setdefault("_refused_commands", []).append(reason)
        logger.warning(
            "%s refused: a simulated runner has no profiler, weights or KV "
            "tensors for it to act on, and no cost model for the time it takes.",
            method,
        )
        return reason

    return refuse


class RefusedControlCommands:
    """The worker methods behind ATOM's utility commands, each refused by name.

    The profiler pair would start a real profiler, and the other seven exist
    only on ATOM's rollout runner, so without them here `busy_loop` skips the
    name and the handler waits forever. Each answers `command:<method>` rather
    than raising, so the engine keeps serving, and records it for
    `refused_commands`.
    """

    start_profiler = _refused("start_profiler")
    stop_profiler = _refused("stop_profiler")
    update_weights = _refused("update_weights")
    update_weights_from_shm = _refused("update_weights_from_shm")
    update_weights_from_ipc = _refused("update_weights_from_ipc")
    release_memory = _refused("release_memory")
    resume_memory = _refused("resume_memory")
    clear_kv_cache = _refused("clear_kv_cache")
    configure_hidden_states = _refused("configure_hidden_states")

    def refused_commands(self) -> tuple[str, ...]:
        """Every refusal this runner answered, in order, for the run summary.

        A method rather than an attribute so the engine can read it over the
        same RPC the commands came in on; a tuple, so the caller gets a copy
        and not the list the runner keeps appending to.
        """
        return tuple(vars(self).get("_refused_commands", ()))


class NonAllocatingRunner(RefusedControlCommands):
    """Overrides that construct a model runner without touching device memory."""

    def _build_and_load_model(self, model_class: Any) -> None:
        """Build nothing and read no checkpoint.

        The base class constructs the model with the default device set to this
        rank's GPU and then fills it from disk. Both halves are skipped here, so
        no weight byte reaches the device and no checkpoint is read.

        A speculative config is refused here, because on the last pipeline rank
        the base goes on to build the drafter on this rank's GPU and load its
        checkpoint. On any rank, it cannot be modelled: a predicted step's reply has
        `draft_token_ids` None and zero `num_rejected`/`num_bonus`, which the
        scheduler accepts as a step that drafted nothing (`Scheduler.postprocess`
        never fills `seq.spec_token_ids`, and reads the zeros), so a caller
        that asked for speculation would get a prediction with it off and no
        error.

        What stands in for the model is installed here instead, on a simulated
        run: the cost backend and the device readings its run file declares.
        """
        self.model = UnbuiltModel(model_class)
        # Cleared on the way out, as both of ATOM's own implementations do: the
        # caller set the default device to this rank's GPU before calling, and
        # the code that runs next is written against a cleared default.
        torch.set_default_device(None)
        if _config_field(self, "speculative_config") is not None:
            raise RunnerRefusal(
                "a predicted step drafts no tokens, so reporting one under a "
                "speculative config would model speculation as off and say "
                "nothing about it; speculative decoding has no step semantics "
                "here yet."
            )
        from atom.compass import run

        run.runner(self)
        logger.info(
            "%s not built and no checkpoint read; no weight bytes on the device.",
            self.model.model_class_name,
        )

    def _maybe_warmup(self) -> None:
        """Skip warmup.

        Warmup runs a forward over a dummy batch against real weights. There are
        none here, so there is nothing to warm. Skipping is also what lets
        construction finish at all: the base class warms the model from inside
        its own `__init__`, and the forward that warmup drives is this class's,
        which refuses.
        """
        return

    def _read_device_memory(self) -> Any:
        """Answer the four device figures off the machine spec.

        ATOM's own reads this rank's allocator and the driver. These four come
        from the device model instead, so every line of budget arithmetic below
        the call runs unchanged against a card this host does not have -- which
        is the whole of the substitution, and the reason no formula is copied
        anywhere in this package.

        The named tuple is ATOM's, imported at call time so this module stays
        importable where there is no driver, and built by keyword: four
        same-typed integers handed over positionally are a reordering nobody
        would see.
        """
        from atom.model_engine.model_runner import DeviceMemoryReadings

        readings = _installed_readings(self)
        return DeviceMemoryReadings(
            free=readings.free.total,
            total=readings.total.total,
            peak_torch=readings.peak_torch.total,
            non_torch=readings.non_torch.total,
        )

    def _estimate_cudagraph_overhead(self) -> int:
        """The graph-pool bytes the budget holds back, off the machine spec.

        ATOM's own derives this from the gap between the allocator's warmup
        peak and its steady state, which is two more device reads, so it is the
        second of the two places `get_num_blocks` touches the device and the
        second thing replaced here.

        The reading installed for it is the one `memory.graph_pool.reserves()`
        produces -- the mirror of ATOM's estimator rather than the measured
        pool, because this is the number that actually reserves and the two
        disagree; `memory.device_readings` refuses the other one by name at the
        call site. Two adjustments inside ATOM's estimator are not mirrored and
        move the block count in opposite directions, and `reserves()`' own
        docstring names both. Both need a drafter, and this runner never has
        one: `_build_and_load_model` refuses a speculative config before the
        base can build a drafter, so construction fails before
        `get_num_blocks` can run.

        `enforce_eager` is reconciled here rather than trusted. ATOM's own
        returns zero under that flag (`_estimate_cudagraph_overhead`) and this
        returns whatever was installed, so correctness would otherwise rest on
        whoever built the reading having passed the same flag this runner is
        configured with. A reading built for a capturing deployment, installed
        on a runner told not to capture, holds back bytes ATOM would not and
        moves the block count with nothing to show for it. The eager branch
        names the config field as the source of its only term, so the reading
        says which deployment it was built for and a disagreement declines.
        """
        reading = _installed_readings(self).cudagraph_overhead
        built_eager = any(term.source == EAGER_SOURCE for term in reading.terms)
        configured_eager = bool(_config_field(self, "enforce_eager"))
        if built_eager != configured_eager:
            raise RunnerRefusal(
                "the installed graph-pool reading was built for a deployment "
                f"with enforce_eager={built_eager} and this runner is "
                f"configured with enforce_eager={configured_eager}; ATOM "
                "reserves nothing for a graph it never captures, so one of "
                "the two would move the block count by the whole of the "
                "reservation without saying so."
            )
        return reading.total

    def get_num_blocks(self) -> dict[str, object]:
        """Size the KV pool with ATOM's own arithmetic over substituted readings.

        The base method is readings and then arithmetic: the utilisation
        budget, the 2% safety margin, `_kv_budget_extra_reserve`, the
        `min(budget, free)` clamp, `plan_pools`, and under pipeline parallelism
        a minimum across stages. None of that is replaced, and none of it is
        written down anywhere in this package. The two methods above are, and
        `super()` does the rest.

        The reply is the base's, forwarded unaltered, and `EngineCore.__init__`
        (`atom/model_engine/engine_core.py`) fixes its four keys:
        `num_kvcache_blocks` and `state_runtime` are subscripted, `pool_entries`
        and `pool_entries_per_req` are taken with a `{}` default. The block
        count goes on to `BlockManager`, which asserts it is greater than zero
        in its `__init__`; the base asserts the same thing first and prints the
        whole budget with it. `state_runtime` is not an opaque value: it is
        handed to `StateRuntime.from_wire`
        (`atom/model_engine/state_runtime.py`), which raises `TypeError` unless
        it is a `Mapping` and `ValueError` unless its key set is exactly
        `{"transfer", "checkpoint_spec"}`. Building that dict here instead of
        forwarding the base's would put all of those failures on the first RPC
        of the engine's life, so it is not built here.

        What this adds is one record and no arithmetic. The count the scheduler
        receives is an integer with nothing attached, and most of what produced
        it is a coefficient somebody wrote down rather than a measurement of
        the card being modelled; `kv_pool_sizing` keeps the count beside the
        readings, so which terms those were is recoverable from the runner and
        not only from a log line somebody kept.

        The decode process of intra-GPU prefill/decode disaggregation is
        refused rather than sized. ATOM's own runner for that answers a block
        count of zero there, because decode imports the pool from prefill and
        owns no device memory (`RapidServeModelRunner.get_num_blocks`), and it
        holds back four safety margins on the prefill side because two
        processes share the card (`_kv_budget_extra_reserve`). Substituting
        readings into the base method reaches neither: this path would hand a
        decode process a pool it does not own.
        The base's `_kv_budget_extra_reserve` of zero is left alone for the
        same reason it is right -- the card these readings describe is a
        dedicated one, and a shared-card reservation would hold bytes back for a
        second tenant that those readings do not have.
        """
        if _config_field(self, "disagg_is_decode"):
            raise RunnerRefusal(
                "this runner does not model the decode process of intra-GPU "
                "prefill/decode disaggregation: that process owns no device "
                "memory and imports the pool from prefill, so its block count "
                "is zero rather than a sizing, and the substituted readings "
                "describe a card with one tenant on it."
            )
        readings = _installed_readings(self)
        reply = super().get_num_blocks()
        self.kv_pool_sizing = SizedKVPool(
            num_kvcache_blocks=int(reply["num_kvcache_blocks"]),
            entries=dict(reply["pool_entries"]),
            readings=readings,
        )
        logger.info("%s", self.kv_pool_sizing.table())
        return reply

    def allocate_kv_cache(self, num_kvcache_blocks: int) -> bool:
        """Record the block count and allocate nothing.

        The block accounting above this -- the pool, the prefix index, eviction,
        preemption -- is integer arithmetic and runs unmodified against the count
        recorded here. Only the tensors behind the blocks are absent.

        An empty registry still goes through ATOM's `set_kv_cache_data`: it is
        the one call that builds the worker-side KV connector, which the engine
        polls for transfer completions.
        """
        from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
        from atom.utils.forward_context import set_kv_cache_data

        kv = _config_field(self, "kv_transfer_config")
        if kv:
            name = KVConnectorFactory.canonical_name(kv.get("kv_connector", "moriio"))
            if name != "compass":
                raise RunnerRefusal(
                    f"kv_connector {name!r} is a real transfer backend; this "
                    "runner simulates only 'compass', and an unset kv_connector "
                    "means 'moriio'."
                )
        set_kv_cache_data({}, self.config, num_blocks=num_kvcache_blocks)
        self.config.num_kvcache_blocks = num_kvcache_blocks
        logger.info(
            "kv cache: %d blocks accounted, 0 bytes allocated.",
            num_kvcache_blocks,
        )
        return True

    def capture_cudagraph(self) -> tuple[float, list[int], int]:
        """Capture nothing, and say so in the three values the caller unpacks.

        The base reaches the model only after it has zeroed device buffers,
        opened a graph memory pool and entered a capture context, so inheriting
        it would put bytes on a device before it discovered there are no
        weights to trace. Nothing is captured here, and the reply says exactly
        that: no seconds spent, no batch sizes captured, no pool bytes held.

        The shape is fixed by `EngineCore.__init__` and
        `DecodeEngineCore.__init__` (`atom/model_engine/engine_core.py`), which
        both write `cap_cost, bs, pool_bytes = ...` and then format the first
        and third as numbers. Returning two values, or None, is not a test
        failure over there: the worker dies on the unpack, or never replies, and
        the caller waits for the rest of its life either way.

        `capture_sizes` and `capture_sizes_np` are deliberately left at the
        eager-fallback values the base set during construction, because the
        attention metadata builder reads them on every step. The reply
        deliberately disagrees with the attribute it preserves: the base sends
        `self.capture_sizes` -- the same `[0]` -- as the second element
        (`ModelRunner.capture_cudagraph`), and this sends `[]`. Both are only
        ever formatted into a log line, and `[]` is the truthful one.

        `EngineCore.__init__` reaches this method only when `not enforce_eager
        and not disagg_is_decode`, so on an eager deployment the override never
        runs at all. Replacing it anyway is the point: a runner must not depend
        on a caller-side flag to stay off the device.
        """
        logger.info("no cudagraph captured: there are no weights to trace.")
        return 0.0, [], 0

    @torch.inference_mode()
    def forward(self, batch: Any) -> Any:
        """Report what a step produced, without running one.

        The reply is the whole of what the scheduler learns from a step, and
        `step_output` holds the three rules that decide it: the tokens belong
        to the previous output-producing batch, the lag is counted in
        output-producing steps rather than in steps, and a batch that samples
        nothing reports its requests with no tokens. Each is wrong in a way the
        scheduler accepts without complaining, so they are stated in one place
        with their reasons rather than inlined here.

        What a replacement owes its callers. These sites broadcast this name --
        `EngineCore._process_engine_step_inner` and
        `DecodeEngineCore._process_engine_step` in
        `atom/model_engine/engine_core.py`, and
        `PPEngineCoreProc._pp_head_step` (which waits and discards the reply)
        and `PPEngineCoreProc._downstream_busy_loop` in
        `atom/model_engine/pp_engine_core.py`. None of them unpacks it; the
        reply is one object, read for its attributes, and the reads are these:

        * `.req_ids`, an iterable of request ids in the batch's own order --
          `PPEngineCoreProc._pp_head_step`. That read is on the head's side of
          the transport, on what `recv_tokens()` returns, not at the
          `_downstream_busy_loop` call site, which only passes the object on.
        * `.token_ids`, `.draft_token_ids`, `.is_deferred_out`, `.logprobs`
          (None or a `dict[int, float]`) -- `Scheduler.postprocess`.
        * `.get_idx(req_id)`, returning a row index or None --
          `Scheduler.postprocess`.
        * `.num_rejected[idx]` and `.num_bonus[idx]`, indexable by that row and
          castable to `int` -- `Scheduler.postprocess`.
        * `.dspark_ell`, either None or a mapping answering `.get(seq.id)` --
          `Scheduler.postprocess`.

        Picklable on top: under pipeline parallelism the last stage sends the
        object whole (`_downstream_busy_loop`) and the head reads it back
        (`_pp_head_step`), each hop through `pickle` in
        `PPStageTransport.send_tokens` and `PPStageTransport.recv_tokens`
        (`atom/distributed/pp_transport.py`). ATOM's own answer is a
        `ScheduledBatchOutput`; the contract a replacement has to meet is these
        reads and the pickle, not that class.

        One decorator, not the base's two. ATOM's own carries
        `torch.inference_mode`, which is kept: the body no longer raises, and
        whatever a cost model eventually evaluates here should build in the
        same context as the step it stands for. It also carries
        `with_eplb_forward_monitor`, which is not. That monitor exists to
        observe how a real forward routed tokens across experts and commits a
        load window from what it saw; a predicted step routes nothing, so the
        window would be fabricated, and acting on one can move experts on a
        device this runner owns none of. A decorator is applied while the
        class body runs, so it would also have to be imported from the engine
        at module scope, which this module does not do.
        `RapidServeModelRunner`, the other runner in this tree that declines
        to own its memory, carries the same one of the two.

        The reply's `predicted_s` is the step's duration: this rank's batch
        priced by the installed cost backend, then the max over its DP group.
        A runner with no backend refuses the step rather than report one that
        took no time. ATOM's `dummy_execution` answers with this reply too, so
        an idle DP rank's dummy batch is priced and joins the group's max.
        """
        if not hasattr(batch, "produces_output"):
            # Not reachable from the engine, which only ever passes a scheduled
            # batch. Kept because it is the one shape this method cannot report
            # from, and because a refusal that did cross the worker boundary
            # would reach the parent as a bare SystemExit -- so the loud
            # failure has to happen on this side of it.
            raise RunnerRefusal(
                "a step is reported from what the scheduler scheduled, and "
                f"{type(batch).__name__} cannot say whether its batch produces "
                "output; there is nothing here to report from."
            )
        backend = _installed_backend(self)
        # Imported at call time, not at module scope: the scheduler so this
        # module stays importable where there is no driver (by the time a step
        # is reported the worker has imported the engine anyway), and the
        # parity package because, at module scope, this module may import no
        # `atom` package besides `atom.compass.runner` and `atom.compass.memory`
        # (`test_only_the_binding_module_reaches_the_engine`).
        from atom.compass.parity import record_step
        from atom.model_engine.scheduler import ScheduledBatchOutput

        record_step(self, batch)

        stream = getattr(self, "_token_stream", None)
        if stream is None:
            # Built on first use rather than in an `__init__`: the base class
            # runs the whole of its own before a subclass body would get
            # control, and `self.config` is what this reads. The ids are vetted
            # against the tokenizer the frontend serves with, loaded the same
            # way.
            from atom.model_engine.llm_engine import _load_tokenizer

            model = _config_field(self, "model")
            fillers = filler_token_ids(
                _load_tokenizer(model, _config_field(self, "trust_remote_code")),
                _config_field(self, "eos_token_id"),
                _config_field(self, "stop_token_ids"),
            )
            if not fillers:
                raise RunnerRefusal(
                    f"the tokenizer of {model!r} has no id that is not a stop id "
                    "and decodes on its own to ASCII letters and digits, so "
                    "there is no token a predicted step can report without "
                    "ending a request or stalling its stream"
                )
            stream = DeferredTokenStream(
                fillers,
                deferred=reports_previous_step(
                    _config_field(self, "pipeline_parallel_size")
                ),
            )
            self._token_stream = stream
        return ScheduledBatchOutput(
            **stream.step(batch),
            predicted_s=_group_step_seconds(self, batch, backend),
        )


def _group_step_seconds(runner: Any, batch: Any, backend: CostBackend) -> float:
    """This rank's step priced from its own batch, then the max over its DP group.

    The `all_reduce(MAX)` stands in for the cross-rank collectives a real
    forward makes, which a predicted one does not run, so every rank leaves the
    step at the group's seconds. With one rank there is no group to exchange
    with.

    A pipeline stage is refused: it runs only its own layers, and a backend
    prices the whole step, so every stage would charge the whole model.
    """
    from atom.compass.runner.projection import batch_view, dp_group, forward_mode

    if int(_config_field(runner, "pipeline_parallel_size") or 1) > 1:
        raise RunnerRefusal(
            "this runner is one stage of a pipeline and runs only its own "
            "layers, but a cost backend prices a whole step; charging each "
            "stage the whole model would count it once per stage"
        )
    mode = forward_mode(batch, runner)
    seconds = backend.estimate(batch_view(batch, mode, runner)).seconds
    if mode.sync is None:
        return seconds
    group_max = torch.tensor([seconds], dtype=torch.float64)
    torch.distributed.all_reduce(
        group_max, op=torch.distributed.ReduceOp.MAX, group=dp_group()
    )
    return group_max.item()
