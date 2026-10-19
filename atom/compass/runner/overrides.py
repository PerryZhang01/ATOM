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
  put it there itself, on the worker side, before it raises. `forward` does:
  it writes what it raises to the file `compass_raised_file` names, which the
  engine reads into its ``ended:`` reason when the `SystemExit` reaches it.

The worker control commands are refused the third way: by a reply. A raise
would end the engine for a command a client sent while serving, and a None
would park it, so `RefusedControlCommands` answers each with its reason,
`command:<method>`, and logs it. `busy_loop` forwards the reason, and
`EngineUtilityHandler` sends it back to the client as the `UTILITY_RESPONSE`
result for every command but `discard_failed_weight_sync`, whose handler
answers `True` once no rank failed; the runner also keeps every reason it
answered for the run summary's refusals.

`RPC_SURFACE` records, for each dispatched name, whether its caller waits for
the reply. The names themselves are not a list anyone typed: they are the ones
`engine_core`, `pp_engine_core` and `engine_utility` broadcast that a
`ModelRunner` answers, and the tests derive that intersection from ATOM's own
source and compare it with this table.
"""

from __future__ import annotations

import collections
import contextlib
import json
import logging
import math
import os
import pickle
from functools import partial, wraps
from typing import Any

import numpy as np
import torch
from torch.overrides import TorchFunctionMode
from torch.utils._device import _device_constructors

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
# The RapidServe names are answered by `RapidServeRpcs`, not by this
# table: `enable_rapidserve` picks `PrefillEngineCore` / `DecodeEngineCore`
# (`LLMEngine.__init__`), those classes broadcast all of them with
# `wait_out=True` without consulting `runner_qualname`, so a simulated
# RapidServe run names the runner that adds them, and `Config` raises
# `ValueError` for `enable_rapidserve=True` with any runner not in
# `RAPIDSERVE_RUNNERS`, this one included.
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
    "async_proc_aggregation": True,  # EngineCore polls it; a simulated run waits for it
    # PPEngineCoreProc waits; ATOM's own answers at once, as a simulated runner
    # holds no real send and the stage loop has already waited for its own.
    "flush_pp_send": True,
    "poll_forward_metrics": False,  # EngineUtilityHandler does not wait
    "release_multimodal_requests": False,  # EngineCore does not wait
    # Refused by `RefusedControlCommands`: ATOM's own starts a real profiler.
    "start_profiler": True,  # EngineUtilityHandler forwards the reply unread
    "stop_profiler": True,  # EngineUtilityHandler forwards the reply unread
}


def unanswered_rpc_names(runner: Any) -> tuple[str, ...]:
    """The dispatched names `getattr` would answer with None for this runner.

    Drawn from the whole surface, which does not fail one way, so this return
    carries no one story about its names. `RPC_SURFACE[name]` records whether
    any caller reads the reply, and this function's only caller in the package
    partitions the result on exactly that before reporting it: a waited name
    parks its caller on an unbounded queue read for the life of the process,
    while a name no caller reads, such as `exit`, parks no one when it is a
    hole.

    Checked where the class is composed instead of being discovered by a
    deployment, because the worker raises on none of them.
    """
    return tuple(name for name in RPC_SURFACE if getattr(runner, name, None) is None)


class RunnerRefusal(RuntimeError):
    """Raised where this runner has no answer and will not invent one."""


def _says_why(forward):
    """`forward`, writing what it raises to the file `compass_raised_file`
    names, when set, before it raises: past the worker boundary the engine
    gets a bare `SystemExit`, and reads the reason there."""

    @wraps(forward)
    def said(self, batch):
        try:
            return forward(self, batch)
        except BaseException as e:
            if path := getattr(self, "compass_raised_file", None):
                with contextlib.suppress(OSError):  # raise `e`, not a failed write
                    path.write_text(f"rank {self.rank} raised {e!r}")
            raise

    return said


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


HOST_SEGMENTS = ("E", "W_pre", "W_launch", "W_post")  # in the order the host runs them
#: The part of prepare after the MTP status wait; an output forward runs it under MTP.
MTP_SEGMENT = "W_inputs"
#: The part of `W_launch` a later pipeline stage runs before it receives its input.
PP_RECV_SEGMENT = "W_recv"
FORWARD_KINDS = ("decode", "prefill", "middle")
#: The format of a simulated run's `dp<r>.cost.jsonl`, its first line.
COST_TABLE_VERSION = 1


def forward_shape(batch: Any) -> tuple[str, tuple]:
    """A forward's kind in `FORWARD_KINDS` and its law features ``(1, tokens, seqs)``."""
    kind = "decode" if batch.total_seqs_num_prefill == 0 else "prefill"
    kind = kind if batch.produces_output() else "middle"
    return kind, (1, batch.total_tokens_num, batch.total_seqs_num)


class HostDeviceRecurrence:
    """One runner's host and device timelines, composed into each forward's charge.

    The host runs `HOST_SEGMENTS` from `now`. A forward's entry waits for the
    device work two forwards back, or on a pipeline stage `stage = (rank, pp)`,
    `pp` forwards back; an output forward, after launch, waits for the previous
    output forward's, or on the last stage for its own, and on any other stage
    for nothing. With `mtp`, a forward's device work is its target's and then
    its draft's, and a single-stage output forward waits before launch, for
    the previous output forward's target; `MTP_SEGMENT` runs after that wait,
    before the launch. With `sync`, every forward leaves no earlier than its
    own device end, as one that synchronizes its stream before it returns
    does. `ends[-1]` is the last forward's device end. `laws[kind][segment]` are coefficients on `(1,
    tokens, seqs)`; `charge` says what an absent one costs. `now` reads the engine LP's clock.
    `segments` holds the seconds the last `charge` charged each segment, 0 for
    one in `skip`.
    """

    def __init__(
        self, laws: dict, skip, now, stage=None, mtp=False, sync=False
    ) -> None:
        self.laws, self.skip, self.now, self.sync = laws, frozenset(skip), now, sync
        ring = 2 if stage is None else stage[1]
        self.ends = collections.deque([-math.inf] * ring, maxlen=ring)  # n-ring..n-1
        self.last = None if stage is None else stage[0] == stage[1] - 1
        self.recv = stage is not None and stage[0] > 0  # a later stage receives
        self.output_end = -math.inf  # target end of the last output forward
        self.mtp = mtp

    def charge(
        self, batch: Any, device_s: float, draft_s: float, refuse: bool = True
    ) -> float:
        """The forward's exit less `now`: `release` of `device_end` and its
        `target_end`."""
        end = self.device_end(batch, device_s, draft_s, refuse)
        return self.release(end, self.target_end)

    def device_end(
        self, batch: Any, device_s: float, draft_s: float, refuse: bool = True
    ) -> float:
        """The end of the forward's device work, launch included, for `device_s`
        of target device time and then `draft_s` of draft, starting no earlier
        than the batch's `compass_pp_release`; `target_end` is the target's
        end, and `release` follows. A batch's
        `compass_pp_host = (called, meta)` runs the host from `called` in place
        of `now`; its launch posts the receive after `PP_RECV_SEGMENT` and
        runs the rest once `meta` is in. With any
        laws and `refuse`, a forward whose kind lacks one for a segment it runs
        is refused, naming each ``<kind>.<segment>``, the state as it was;
        without `refuse` that segment is 0. A segment in `skip` needs none, a
        middle chunk runs no `W_post`, and only an output forward under `mtp`
        runs `MTP_SEGMENT`; a stage after the first runs `PP_RECV_SEGMENT`."""
        kind, x = forward_shape(batch)
        law = self.laws.get(kind, {})
        ran = HOST_SEGMENTS[: 3 if kind == "middle" else 4]
        ran += (MTP_SEGMENT,) if self.mtp and kind != "middle" else ()
        ran += (PP_RECV_SEGMENT,) if self.recv else ()
        missing = [s for s in ran if s not in {*law, *self.skip}]
        if refuse and self.laws and missing:
            from atom.compass.backends.resolver import CostRefused, Refusal

            named = ", ".join(f"{kind}.{s}" for s in missing)
            raise CostRefused(kind, [Refusal("host", f"no law for {named}")])
        now = self.now()
        called, meta = getattr(batch, "compass_pp_host", (now, -math.inf))
        prepared, self.segments = self.prepare_end(called, kind, x)
        launch, post = self.segments["W_launch"], self.segments["W_post"]
        host = max(prepared, meta)
        output = batch.produces_output()
        launch_end = max(prepared, meta - self.segments[PP_RECV_SEGMENT]) + launch
        release = getattr(batch, "compass_pp_release", -math.inf)
        self.target_end = max(self.ends[-1], host, release) + device_s
        self._held = now, launch_end, output, post
        return max(self.target_end + draft_s, launch_end)

    def prepare_end(self, called: float, kind: str, x: tuple) -> tuple[float, dict]:
        """A forward's prepare end, with `MTP_SEGMENT`, and each segment's seconds,
        `PP_RECV_SEGMENT`'s too."""
        law = self.laws.get(kind, {})
        seg = {
            s: float(np.dot(law.get(s, (0, 0, 0)), x)) * (s not in self.skip)
            for s in (*HOST_SEGMENTS, MTP_SEGMENT, PP_RECV_SEGMENT)
        }
        host = max(called + seg["E"], self.ends[0]) + seg["W_pre"]
        if self.mtp and kind != "middle":  # output_end stays -inf on a PP stage
            host = max(host, self.output_end)
        return host + seg[MTP_SEGMENT], seg

    def release(self, end: float, target_end: float) -> float:
        """The exit less `now` of the forward `device_end` priced, its device
        work ending at `end` and its target's at `target_end`, no earlier than
        it said."""
        now, launch_end, output, post = self._held
        self.ends.append(end)
        released = launch_end
        if self.sync:
            released = max(launch_end, end)
        elif output:
            if self.last is None:
                released, self.output_end = max(launch_end, self.output_end), target_end
            elif self.last:
                released = max(launch_end, target_end)
        return released + post - now


def schedulable_ladder(config: Any) -> list[int]:
    """The capture sizes the scheduler can produce, ascending: ATOM's capture
    bound, each sequence charged one token plus its drafts.
    `max_schedulable_decode_bs` holds it; importing that would pull the engine,
    and its kernels, into a driver-free process."""
    spec = getattr(config, "speculative_config", None)
    q = 1 + (spec.num_speculative_tokens if spec else 0)
    bound = min(config.max_num_seqs, config.max_num_batched_tokens // q)
    return sorted(s for s in config.capture_sizes if s <= bound)


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


class HostEvent:
    """Stands in for a device event on a host start: nothing to order or wait on."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record(self, stream: Any = None) -> None:
        pass

    def synchronize(self) -> None:
        pass


def start_on_host(runner: Any, rank: int, config: Any) -> None:
    """Start a runner on the host: no device set, its groups on gloo.

    Replaces ATOM's `_setup_device_and_distributed`, which
    `ModelRunner.__init__` calls before it builds any buffer, so every buffer
    the rest of `__init__` builds on `runner.device` lands on the host. Three
    touches there do not follow the device: `torch.cuda.Stream` refuses `cpu`,
    `torch.cuda.Event` records on a device, and `CpuGpuBuffer` pins its host
    half, a driver allocation. All are replaced for the life of the worker
    process, which hosts this runner and nothing else.

    The groups are the ones ATOM's own call builds -- TP, DCP, PCP, PP, DP and
    EP over `rank` of the engine's TP x PCP workers, each pipeline stage's
    after the stages before it -- on gloo and with an empty custom group
    configuration, under which aiter builds no device communicator for any
    group. A simulated TP width is refused: it starts fewer workers than the
    TP group `apply_simulated_tp` then reports on a device group.
    """
    pc = config.parallel_config
    if config.tp_world_size != config.tensor_parallel_size:
        raise RunnerRefusal(
            f"-tp {config.tensor_parallel_size} starts {config.tp_world_size} "
            "workers under --fake-eplb, and a runner on the host builds its TP "
            "group from the ranks that start."
        )
    from aiter.dist.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from aiter.dist.utils import get_distributed_init_method

    from atom.utils import CpuGpuBuffer

    pinned = CpuGpuBuffer.__init__

    def unpinned(buffer: Any, *size: Any, pin_memory: bool = True, **kw: Any):
        pinned(buffer, *size, pin_memory=False, **kw)

    CpuGpuBuffer.__init__ = unpinned
    torch.cuda.Stream = HostStream
    torch.cuda.Event = HostEvent
    runner.device = torch.device("cpu")
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = str(config.port)
    method = get_distributed_init_method(
        pc.data_parallel_master_ip, pc.data_parallel_base_port
    )
    tp, pcp = config.tensor_parallel_size, config.prefill_context_parallel_size
    pp = config.pipeline_parallel_size
    # This worker's rank among its DP rank's stages, as ATOM numbers it.
    rank += (pc.pipeline_parallel_rank if pp > 1 else 0) * tp * pcp
    if pc._managed_distributed_store and not torch.distributed.is_initialized():
        # The engine core manager already listens on the base port; join its
        # store as a client, as ModelRunner does, and aiter reuses the group.
        store = torch.distributed.TCPStore(
            pc.data_parallel_master_ip, pc.data_parallel_base_port, is_master=False
        )
        torch.distributed.init_process_group(
            backend="gloo",
            store=store,
            rank=pc.data_parallel_rank * pp * tp * pcp + rank,
            world_size=pc.data_parallel_size * pp * tp * pcp,
        )
    init_distributed_environment(
        world_size=pp * tp * pcp,
        rank=rank,
        distributed_init_method=method,
        backend="gloo",
        local_rank=0,
        data_parallel_size=pc.data_parallel_size,
        data_parallel_rank=pc.data_parallel_rank,
    )
    initialize_model_parallel(
        tp,
        pp,
        decode_context_model_parallel_size=config.decode_context_parallel_size,
        data_parallel_size=pc.data_parallel_size,
        prefill_context_model_parallel_size=pcp,
        custom_group_config={},
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


class FactoriesOnMeta(TorchFunctionMode):
    """Puts what a factory function makes on the meta device, a named device too.

    `torch.device("meta")` fills in only a device the call leaves out, so a
    module that names one -- ATOM's attention puts its KV scale on this rank's
    GPU -- would still allocate there and bring up a device context. The
    functions are the ones `torch.device` itself redirects, `torch.zeros` and
    `torch.tensor` among them; a `*_like` or `Tensor.new_*` call keeps the
    device it names.
    """

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _device_constructors():
            kwargs["device"] = "meta"
        return func(*args, **kwargs)


def process_weights_after_loading(tree: torch.nn.Module) -> None:
    """Run on a meta tree the post-load hooks `load_model` runs after a load.

    The same walk, parent first: each module's own hook, then its quant
    method's. The hooks re-lay weights (an MXFP4 MoE pads its expert scales),
    so the storages they leave are the ones a real start holds. The loader's
    `init_prepare_finalize` is not run: it sets up an MoE's dispatch and holds
    no weight. A hook that fails on the meta device refuses the start by name.
    """
    if not isinstance(tree, torch.nn.Module):
        return
    from atom.model_ops.base_config import QuantizeMethodBase

    for name, module in tree.named_modules():
        hooks = []
        if hasattr(module, "process_weights_after_loading"):
            hooks.append((module, module.process_weights_after_loading))
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            hooks.append(
                (
                    quant_method,
                    partial(quant_method.process_weights_after_loading, module),
                )
            )
        for owner, hook in hooks:
            try:
                hook()
            except Exception as e:
                raise RunnerRefusal(
                    f"{type(owner).__name__}.process_weights_after_loading on "
                    f"{name or type(tree).__name__!r} cannot run on the meta "
                    f"device ({e!r}), so the weights it leaves are unknown."
                ) from e


def drafts_on_meta() -> None:
    """Make every drafter flavor build its model on the meta device, and skip
    its checkpoint load and its hooks on the target model, for the life of the
    worker process.

    `ModelRunner.__init__` builds the drafter on this rank's device and loads it
    right after `_build_and_load_model`, with no override point between. The
    drafter's own buffers stay on the host; only the draft model is meta. Every
    subclass of `Drafter` is patched, at any depth; a hook `Drafter` no longer
    defines is refused by name, since the flavors would then build on this
    rank's device and read the checkpoint.
    """
    from atom.spec_decode.drafter import Drafter

    hooks = ("_build_draft_model", "load_model", "arm_aux_capture")
    if missing := [h for h in hooks if h not in vars(Drafter)]:
        raise RunnerRefusal(
            f"Drafter no longer defines {', '.join(missing)}, which a simulated "
            "run patches to build the drafter on meta and read no checkpoint"
        )

    def on_meta(build):
        @wraps(build)
        def built(self, model_class):
            with FactoriesOnMeta():
                return build(self, model_class)

        return built

    classes = [Drafter]
    for cls in classes:  # grows as it goes: every subclass, at any depth
        classes += cls.__subclasses__()
        if build := vars(cls).get("_build_draft_model"):
            cls._build_draft_model = on_meta(build)
        for name in ("load_model", "arm_aux_capture"):
            if name in vars(cls):
                setattr(cls, name, lambda self, target: None)


class UnbuiltModel(torch.nn.Module):
    """Stands in for the model a non-allocating runner never loads.

    It holds the model's module tree as `tree`, built on the meta device, so
    ATOM can walk its attention modules to size the KV pool while no parameter
    or buffer has a byte on any device. Calling it raises instead of returning
    something a caller could mistake for a forward pass.
    """

    def __init__(self, model_class: Any) -> None:
        super().__init__()
        self.model_class_name = getattr(model_class, "__name__", repr(model_class))

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise RunnerRefusal(
            f"{self.model_class_name} was never loaded: this runner predicts what a "
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

    The profiler pair would start a real profiler, and the others exist only
    on ATOM's rollout runner: without them here `busy_loop` skips a name sent
    by `call_func` and the handler waits forever, and answers one sent over
    the all-rank RPC with a missing-method error. Each answers
    `command:<method>` rather than raising, so the engine keeps serving, and
    records it for `refused_commands`. A refused weight update still fences
    serving until a sync is finished, as ATOM does for any update.
    """

    start_profiler = _refused("start_profiler")
    stop_profiler = _refused("stop_profiler")
    update_weights = _refused("update_weights")
    update_weights_from_shm = _refused("update_weights_from_shm")
    update_weights_from_ipc = _refused("update_weights_from_ipc")
    discard_failed_weight_sync = _refused("discard_failed_weight_sync")
    release_memory = _refused("release_memory")
    resume_memory = _refused("resume_memory")
    clear_kv_cache = _refused("clear_kv_cache")
    configure_hidden_states = _refused("configure_hidden_states")

    def refused_commands(self) -> tuple[str, ...]:
        """Every refusal this runner answered, in order, for the run summary:
        each refused command, and each step no cost source priced.

        A method rather than an attribute so the engine can read it over the
        same RPC the commands came in on; a tuple, so the caller gets a copy
        and not the list the runner keeps appending to.
        """
        return tuple(vars(self).get("_refused_commands", ()))


class PricedSteps:
    """What a runner's steps cost, read by the run summary over the worker RPC."""

    def priced_steps(self) -> dict[str, Any]:
        """This rank's own steps, folded into one `ProvenanceMix` as each was
        priced, before the max over the DP group, with the steps and seconds
        of each of `GROUPS` apart, and the steps a source extrapolated and
        those a fallback priced after a source declined. A dict even before
        the first step, since a None reply parks the engine that asks.
        """
        from atom.compass.backends.cost import GROUPS, ProvenanceMix, group

        mix = vars(self).get("_priced") or ProvenanceMix()
        steps = {group(*k): n for k, n in mix.steps_by_method().items()}
        seconds = {group(*k): s for k, s in mix.seconds_by_method().items()}
        return {
            "steps": mix.steps,
            "seconds": mix.seconds,
            "extrapolated_steps": mix.extrapolated_steps,
            # A priced step that carries refusals fell through to a later source.
            "fallback_steps": mix.fallback_steps,
            **{f"{g}_steps": steps.get(g, 0) for g in GROUPS},
            **{f"{g}_seconds": seconds.get(g, 0.0) for g in GROUPS},
            "treatments": dict(mix.treatments),
        }


class PredictedMemory:
    """The memory a runner predicts, read by the engine's hand-in over the worker RPC."""

    def predicted_memory(self) -> dict[str, Any]:
        """`compass_memory` as a run installed it; a None reply parks the engine."""
        return vars(self).get("compass_memory", {})


class StepGridBundle:
    """The step grid bundle a runner prices with, read by the engine's hand-in."""

    def step_grid_bundle(self) -> dict[str, Any]:
        """`compass_step_grid` as a step grid run installed it; empty before
        one is, since a None reply parks the engine."""
        return vars(self).get("compass_step_grid", {})


class PipelineFrames:
    """A pipeline stage's send to the next, and the frames its stage loop
    moves between the stages' TP-rank-0 workers."""

    def _pp_send(self, batch: Any, seconds: float) -> tuple | None:
        """What this stage's forward sends the next stage, None on the last:
        when its device work ends, in seconds from the call; the `W_post` the
        stage loop charges after its send, left out of `predicted_s`; and its parts as
        ``(carrier, bytes)`` in the order the next receives them, the size and
        the pickled metadata on gloo, then each tensor on RCCL, as ATOM splits
        `_pp_output`'s `_PP_PROXY_KEYS`; a key it does not size is refused."""
        pp = int(_config_field(self, "pipeline_parallel_size") or 1)
        if pp == 1 or self.config.parallel_config.pipeline_parallel_rank == pp - 1:
            return None
        from atom.distributed import pp_comm

        out = self._pp_output(batch.total_tokens_num)
        if unsized := set(pp_comm._PP_PROXY_KEYS) - out.keys():
            raise RuntimeError(f"Compass cannot size ATOM's PP keys {sorted(unsized)}")
        sent = {k: out[k] for k in pp_comm._PP_PROXY_KEYS if out[k] is not None}
        # A "meta" device pickles to the length of ATOM's "cuda".
        meta, device_tensors = pp_comm._split_tensor_dict(sent)
        tensors = [("rccl", t.nbytes) for t in device_tensors]
        host = getattr(self, "compass_host", None)
        post = 0.0
        if isinstance(host, HostDeviceRecurrence) and self.rank == 0:
            seconds, post = host.ends[-1] - host.now(), host.segments["W_post"]
        return seconds, post, [("gloo", 8), ("gloo", len(pickle.dumps(meta))), *tensors]

    def _pp_output(self, n: int) -> dict:
        """Each key a non-last stage's output can carry to ATOM's PP send, a
        meta tensor for `n` tokens, or None where this stage does not carry it."""
        self._setup_pp_shared_indexer()
        meta, on = partial(torch.empty, device="meta"), self._pp_send_needs_sparse
        x = meta(n, self.hf_text_config.hidden_size, dtype=self.config.torch_dtype)
        sparse = meta(n * self._pp_index_topk, dtype=torch.int32) if on else None
        return {"hidden_states": x, "residual": x, "sparse_kv_indices": sparse}

    def pp_recv_posted(self, called: float, kind: str, x: tuple) -> tuple:
        """Where this stage posts its ``pp_data`` receive, its forward's prepare
        end and `PP_RECV_SEGMENT`, and its device end; the call and -inf with no
        host recurrence."""
        host = getattr(self, "compass_host", None)
        if not isinstance(host, HostDeviceRecurrence):
            return called, -math.inf
        end, seg = host.prepare_end(called, kind, x)
        return end + seg[PP_RECV_SEGMENT], host.ends[-1]

    def send_pp_frame(self, kind: str, frame: dict) -> bool:
        """Post one stage-loop frame to the next stage (``pp_data``) or the one
        before (``pp_ack``), from TP rank 0, without waiting for it."""
        if self.rank == 0:
            from aiter.dist.parallel_state import get_pp_group

            from atom.distributed.pp_comm import _async_send_object

            pp = get_pp_group()
            dst = pp.ranks[pp.rank_in_group + (1 if kind == "pp_data" else -1)]
            # Each in-flight send's work and payload stay alive until it completes.
            live = [
                w for w in getattr(self, "_pp_frames", []) if not w.work.is_completed()
            ]
            self._pp_frames = live + _async_send_object(frame, dst, pp.cpu_group)
        return True

    def recv_pp_frame(self, kind: str) -> Any:
        """Read one released frame from the stage before (``pp_data``) or the
        next (``pp_ack``), on TP rank 0."""
        if self.rank != 0:
            return True
        from aiter.dist.parallel_state import get_pp_group

        pp = get_pp_group()
        return pp.recv_object(pp.rank_in_group + (-1 if kind == "pp_data" else 1))


class OffloadDecisions:
    """The offload decisions each dispatch hands the worker, kept for the
    parity record before ATOM's own dispatch runs."""

    def process_kvconnector_output(self, meta: Any) -> None:
        from atom.compass.parity import record_kv

        record_kv(self, meta)
        return super().process_kvconnector_output(meta)


class NonAllocatingRunner(
    RefusedControlCommands,
    PricedSteps,
    PredictedMemory,
    StepGridBundle,
    PipelineFrames,
    OffloadDecisions,
):
    """Overrides that construct a model runner without touching device memory."""

    def _build_and_load_model(self, model_class: Any) -> None:
        """Build the module tree on the meta device and read no checkpoint.

        The base class constructs the model with the default device set to this
        rank's GPU and then fills it from disk. Here it is built on the meta
        device, shapes and no bytes, a factory function's tensor on a named
        device included, because ATOM sizes the KV pool by walking its attention
        modules; the load is skipped, so no checkpoint is read.

        Under a speculative config the base goes on to build the drafter and
        load it from the checkpoint; `drafts_on_meta` makes that build meta and
        skips the load. A predicted step's accepted drafts are drawn from the
        config's forced acceptance (`DeferredTokenStream`), so a config that
        declares none is refused, as is one over more than one pipeline stage,
        on which ATOM's `prepare_input_ids` raises `NotImplementedError`.

        What stands in for the model is installed here instead, on a simulated
        run: the cost backend and the device readings its run file declares.
        """
        self.model = UnbuiltModel(model_class)
        # Cleared on the way out, as both of ATOM's own implementations do: the
        # caller set the default device to this rank's GPU before calling, and
        # the code that runs next is written against a cleared default.
        torch.set_default_device(None)
        spec = _config_field(self, "speculative_config")
        if spec is not None:
            if spec.synthetic_acceptance_rates is None:
                raise RunnerRefusal(
                    "a predicted step draws its accepted drafts from a declared "
                    "acceptance, and this speculative config declares none; pass "
                    "--spec-decode-acceptance-length, -rate or -rates"
                )
            if int(_config_field(self, "pipeline_parallel_size") or 1) > 1:
                raise RunnerRefusal(
                    "ATOM's prepare_input_ids raises NotImplementedError for "
                    "pipeline parallel + speculative decode"
                )
            drafts_on_meta()
        with FactoriesOnMeta():
            self.model.tree = model_class(self.config)
            process_weights_after_loading(self.model.tree)
        from atom.compass import run

        run.runner(self, self.model.tree)
        logger.info(
            "%s built on meta and no checkpoint read; no weight bytes on a device.",
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
        docstring names both. Both need a drafter: a run under one refuses the
        piecewise branch, and ATOM refuses DSpark's confidence schedule beside
        the forced acceptance this runner requires.

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
        decode process a pool it does not own. `CompassRapidServeModelRunner`
        answers decode and inherits that reserve for prefill; this class adds
        no reserve of its own.
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
        polls for transfer completions. An `lmcache_mp` one, alone or in a
        `multi`, is built as `compass`'s worker half, which moves nothing.
        """
        from atom.compass.kv.lmcache_mp import substitute
        from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
        from atom.utils.forward_context import set_kv_cache_data

        names = []
        if _config_field(self, "kv_transfer_config"):
            names = [
                n for n, _ in KVConnectorFactory.leaf_connectors(self.config) or ()
            ]
        for name in names:
            if name not in ("compass", "lmcache_mp"):
                raise RunnerRefusal(
                    f"kv_connector {name!r} is a real transfer backend; this "
                    "runner simulates only 'compass' and 'lmcache_mp', and an "
                    "unset kv_connector means 'moriio'."
                )
        if "lmcache_mp" in names:
            substitute()
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

        `capture_sizes` and `capture_sizes_np` are set to the ladder the base
        leaves behind: the declared `config.capture_sizes` narrowed to what
        the scheduler can produce, ascending. `ForwardMode.decide` reads the
        rung of every step off `capture_sizes_np`, so leaving the
        construction-time `[0]` would price every decode step as eager. An empty narrowed ladder is refused: ATOM's capture asserts
        on it, so that deployment never starts. The reply still sends `[]` for the batch sizes, since no
        graph exists; the caller only formats it into a log line.

        `EngineCore.__init__` reaches this method only when `not enforce_eager
        and not disagg_is_decode`, so on an eager deployment the override never
        runs at all. Replacing it anyway is the point: a runner must not depend
        on a caller-side flag to stay off the device.
        """
        self.capture_sizes = schedulable_ladder(self.config)
        if not self.capture_sizes:
            bound = min(self.config.max_num_seqs, self.config.max_num_batched_tokens)
            raise RunnerRefusal(
                f"no cudagraph capture size in {self.config.capture_sizes} is "
                f"schedulable at bound {bound}; ATOM's capture refuses to "
                "start this deployment, so there is no ladder to price"
            )
        self.capture_sizes_np = np.asarray(self.capture_sizes, dtype=np.int32)
        logger.info(
            "no cudagraph captured: there are no weights to trace; steps "
            "dispatch on the ladder %s.",
            self.capture_sizes,
        )
        return 0.0, [], 0

    @_says_why
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
        priced by the installed cost backend, then its DP group target, on TP
        rank 0; the engine reads no other rank's reply, and it reports 0 s.
        Its `out_s` and `group_out` are `_group_step_seconds`'s `_out_s` and
        `_group_out`, which `clock.charge` advances after the return.
        A runner with no backend refuses the step rather than report one that
        took no time. ATOM's `dummy_execution` answers with this reply too, so
        an idle DP rank's dummy batch is priced and joins the group's targets.
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

        mode = record_step(self, batch)
        record = getattr(self, "_parity_record", None)
        if record is not None and record.step == 1:
            # The record was just created, replacing an earlier run's; so is
            # the cost table, before anything this step does can stop the run.
            record.path.with_suffix(".cost.jsonl").write_text(
                json.dumps({"version": COST_TABLE_VERSION}) + "\n"
            )

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
            spec, draws = _config_field(self, "speculative_config"), {}
            if spec is not None:  # one seeded stream per DP rank
                draws = {
                    "rates": spec.synthetic_acceptance_rates,
                    "seed": self.config.parallel_config.data_parallel_rank,
                }
            stream = DeferredTokenStream(
                fillers,
                deferred=reports_previous_step(
                    _config_field(self, "pipeline_parallel_size")
                ),
                **draws,
            )
            self._token_stream = stream
        tokens = stream.step(batch)
        seconds = _group_step_seconds(self, batch, backend, mode)
        out = ScheduledBatchOutput(
            **tokens,
            predicted_s=seconds,
            group_targets=getattr(self, "_group_targets", None),
        )
        out.out_s = getattr(self, "_out_s", None)
        out.group_out = getattr(self, "_group_out", None)
        out.pp_send = self._pp_send(batch, seconds)
        if out.pp_send is not None:
            out.predicted_s -= out.pp_send[1]
        return out


def _group_step_seconds(
    runner: Any, batch: Any, backend: CostBackend, mode: Any = None
) -> float:
    """This rank's step priced from its own batch, then its DP group target.

    Only TP rank 0 prices; any other TP rank of the engine charges 0 s.

    `mode` is the step's `ForwardMode` when the step record already settled
    it; without one it is settled here, which runs the DP collective.

    In a run, `compass_host` composes the price, the target's device seconds
    and then the draft's (`StepCost.target_seconds`, `draft_seconds`), with
    host time. Under DP, a `HostDeviceRecurrence`'s device end, and its output
    mark (the device end, or under MTP the target end), are first replaced by
    the group's latest, an `all_reduce(MAX)` standing in
    for a real forward's cross-rank collectives, and every rank leaves at its
    own charge. With no
    `compass_host`, a rank with output, not the DP dummy, leaves at the group's
    max, any other at its own seconds. `_group_targets` holds all,
    by rank, or None with one rank. `_out_s` is the host's ``out_s`` for this
    step, the seconds from its return to the engine's next input pull, or
    None; `_group_out` holds each rank's target plus its ``out_s``, for the
    ranks with one, or None.

    A pipeline stage runs only its own layers, so it is refused unless its
    backend states the `stage_share` it prices; any other backend prices the
    whole step, and every stage would charge the whole model.

    A step no cost source answers costs 0 s on this rank and still counts as
    a priced step; its reason, ``cost:`` and what each source said, joins the
    refusals the summary reads. With `compass_abort` set it raises instead,
    which ends the worker. Under a speculative config, a step whose price tags
    no `draft` term is refused the same way, since the drafter runs after
    every target forward. A forward `compass_host` refuses takes the same path,
    charged with each segment it named at 0; a step both refuse has one reason.

    On the rank that keeps the parity record, each step also appends its cost
    line to `dp<r>.cost.jsonl` beside it: the record's `step`; with a
    `compass_host` that reports them, each of `HOST_SEGMENTS` and
    `MTP_SEGMENT` and `PP_RECV_SEGMENT`, the part of `W_launch` before a
    later stage's receive, in seconds, each 0.0 where the law has none, and `skipped`,
    those its `skip` charged as 0; `device_s` and `draft_s`, the backend's
    target and draft price; `charge_s`, those composed with host time (their
    sum with no host);
    `group_s`, its DP group target; `terms`, each one's `name`, `seconds`
    and its provenance's `method`, `validated` and `source`; and `refused`,
    the ``cost:`` reason or None. Above them, `forward` starts the file with
    `{"version": COST_TABLE_VERSION}` as the record starts, replacing an
    earlier run's file.
    """
    from atom.compass.runner.projection import batch_view, dp_group, forward_mode

    if runner.rank != 0:
        # The engine reads TP rank 0's reply only, and the other ranks'
        # collectives are inside the step that rank prices.
        return 0.0
    pp = int(_config_field(runner, "pipeline_parallel_size") or 1)
    if pp > 1 and getattr(backend, "stage_share", None) is None:
        raise RunnerRefusal(
            "this runner is one stage of a pipeline and runs only its own "
            f"layers, but {type(backend).__name__} prices a whole step; charging "
            "each stage the whole model would count it once per stage"
        )
    from atom.compass.backends.cost import ProvenanceMix, fold_seconds
    from atom.compass.backends.resolver import CostRefused, Refusal

    if mode is None:
        mode = forward_mode(batch, runner)
    mix = vars(runner).setdefault("_priced", ProvenanceMix())
    spec = getattr(runner.config, "speculative_config", None)
    terms, refused = (), None
    try:
        cost = backend.price(batch_view(batch, mode, runner))
        if spec and not any(t.draft for t in cost.terms):
            name = type(backend).__name__
            raise CostRefused(batch, [Refusal(name, "prices no draft")])
    except CostRefused as refusal:
        if getattr(runner, "compass_abort", False):
            raise
        mix.steps += 1
        said = "; ".join(f"{r.source}: {r.reason}" for r in refusal.declined)
        refused = f"cost:{said or 'no cost source was consulted'}"
        device_s = draft_s = 0.0
    else:
        mix.record(cost)
        device_s, draft_s, terms = cost.target_seconds, cost.draft_seconds, cost.terms
    host = getattr(runner, "compass_host", None)
    group = dp_group() if mode.sync is not None else None
    charge = getattr(host, "charge", None)
    if group is not None and isinstance(host, HostDeviceRecurrence):

        def charge(batch, device_s, draft_s, refuse=True):
            end = host.device_end(batch, device_s, draft_s, refuse)
            # The output end: the group's device end, or under MTP its target end.
            mark = host.target_end if host.mtp else end
            t = torch.tensor([end, mark], dtype=torch.float64)
            torch.distributed.all_reduce(t, torch.distributed.ReduceOp.MAX, group)
            return host.release(*t.tolist())

    try:
        total = fold_seconds((device_s, draft_s))
        charge_s = total if host is None else charge(batch, device_s, draft_s)
    except CostRefused as refusal:
        if getattr(runner, "compass_abort", False):
            raise
        said = "; ".join(f"{r.source}: {r.reason}" for r in refusal.declined)
        refused = (f"{refused}; " if refused else "cost:") + said
        charge_s = charge(batch, device_s, draft_s, refuse=False)
    if refused:
        vars(runner).setdefault("_refused_commands", []).append(refused)
    group_s, targets, pulls = charge_s, None, None
    out_s = getattr(host, "out_s", None)
    if group is not None:
        waits = host is None and batch.produces_output() and not batch.is_dummy_run
        out = math.nan if out_s is None else out_s
        mine = torch.tensor([charge_s, waits, out], dtype=torch.float64)
        rows = [torch.empty_like(mine) for _ in range(group.size())]
        torch.distributed.all_gather(rows, mine, group=group)
        top = max(r[0].item() for r in rows)
        rows = [r.tolist() for r in rows]
        targets = tuple(top if w else s for s, w, _ in rows)
        ends = zip(targets, (o for *_, o in rows))
        pulls = tuple(t + o for t, o in ends if not math.isnan(o)) or None
        group_s = targets[group.rank()]
    vars(runner)["_group_targets"] = targets
    vars(runner)["_out_s"], vars(runner)["_group_out"] = out_s, pulls
    record = getattr(runner, "_parity_record", None)
    if record is None:
        return group_s
    step = record.step - 1  # the step `record_step` just recorded
    segments = {}
    if hasattr(host, "segments"):
        skipped = [s for s in host.segments if s in host.skip]
        segments = {**host.segments, "skipped": skipped}
    line = {
        "step": step,
        **segments,
        "device_s": device_s,
        "draft_s": draft_s,
        "charge_s": charge_s,
        "group_s": group_s,
        "terms": [
            {
                "name": t.name,
                "seconds": t.seconds,
                "method": t.provenance.method.value,
                "validated": t.provenance.validated,
                "source": t.provenance.source,
            }
            for t in terms
        ],
        "refused": refused,
    }
    with record.path.with_suffix(".cost.jsonl").open("a") as f:
        f.write(json.dumps(line) + "\n")
    return group_s


class SampledIds(list):
    """A prefill forward's reply: a sampled token id per request, and as
    attributes the step's price that `clock.charge` reads off a reply."""


class RapidServeRpcs:
    """The RPCs RapidServe's engine cores wait on (`atom.config.RAPIDSERVE_RPCS`),
    on a runner that holds no weights and no KV tensors; it goes before
    `NonAllocatingRunner` in the bases.

    The IPC exports hand over nothing, the imports take nothing in, and no
    stream is made. Decode sizes no pool: it answers as ATOM's
    `RapidServeModelRunner.get_num_blocks` does and allocates nothing, then
    takes prefill's block count with the pool's import. Prefill's pool is
    sized with that class's shared-card reserve, which the composed runner
    inherits from it.
    """

    def get_num_blocks(self) -> dict[str, object]:
        if self.config.disagg_is_decode:
            from atom.model_engine.model_runner import RapidServeModelRunner

            return RapidServeModelRunner.get_num_blocks(self)
        return super().get_num_blocks()

    def allocate_kv_cache(self, num_kvcache_blocks: int) -> bool:
        if self.config.disagg_is_decode:
            return True
        return super().allocate_kv_cache(num_kvcache_blocks)

    def import_kv_cache_ipc_handle(self, paths: Any, num_kvcache_blocks: int) -> bool:
        return super().allocate_kv_cache(num_kvcache_blocks)

    def export_kv_cache_ipc_handle(self) -> list:
        return []

    def export_model_weight_ipc_handles(self) -> list:
        return []

    def import_model_weight_ipc_handles(self, paths: Any) -> bool:
        return True

    def create_prefill_stream_pool(self) -> bool:
        return True

    def create_decode_stream_pool(self) -> bool:
        return True

    def prefill_forward(self, batch: Any) -> SampledIds:
        """`forward`'s step, replied as ATOM's prefill replies: a token id per
        request in batch order, picked by request id as `DeferredTokenStream`
        picks them, since prefill reports its tokens at once. The step is the
        class's `forward`: prefill's instance `forward` is this method."""
        out = super().forward(batch)
        ids = self._token_stream.token_ids
        sampled = SampledIds(ids[int(r) % len(ids)] for r in batch.req_ids)
        for name in ("predicted_s", "out_s", "group_targets", "group_out"):
            setattr(sampled, name, getattr(out, name, None))
        return sampled
