# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The runner class that `runner_qualname` names."""

from __future__ import annotations

from atom.compass.runner.overrides import (
    RPC_SURFACE,
    NonAllocatingRunner,
    RapidServeRpcs,
    RunnerRefusal,
    start_on_host,
    unanswered_rpc_names,
)
from atom.model_engine.model_runner import ModelRunner


class CompassModelRunner(NonAllocatingRunner, ModelRunner):
    """A `ModelRunner` that constructs without weights, KV tensors or a step.

    Everything the engine does around a step -- the scheduler, admission, the
    block manager, the prefix index -- is ATOM's own and runs unchanged. What
    this class removes is the weights, the KV tensors, the graph capture and the
    step itself, by overriding the six methods that own them; see `overrides`,
    which holds the bodies and says why each one does what it does. Through
    `NonAllocatingRunner` it also inherits `RefusedControlCommands`, which
    answers the worker methods behind ATOM's utility commands with a refusal.

    Construction sets no device and puts nothing on one: `start_on_host`
    replaces the base's device and group setup, so `self.device` is the host
    and the groups are on gloo. What stays resident is host memory, almost all
    of it the base's forward-vars ring from `allocate_forward_vars`, whose
    dominant term is a `max_num_batched_tokens` by `hidden_size` output
    buffer; the rest is the ring's events, the attention metadata builder and
    the expert-load-balancing runtime. That dominant term is allocated once
    however deep the pipeline is: a ring slot clones the staging buffers and
    shares the one output buffer.

    Two named attributes on the runner hold that ring for the life of the
    process: `forward_vars`, the dict `allocate_forward_vars` builds, and
    `_fv_ring`, the list of per-slot dicts built from it. Both are the base's.
    What this class adds is no tensor at all: `model`, a module registering no
    parameter and no buffer; `_token_stream`, the deferral bookkeeping
    `forward` builds on first use; and `kv_pool_sizing`, the block count
    `get_num_blocks` answered, kept beside the readings it was sized from.
    Those readings come from `atom.compass.memory`, whose whole import closure
    reaches no tensor library, and `install_device_readings` puts them on the
    runner as `compass_readings`. It also sets `config.num_kvcache_blocks`,
    which is a count and not a buffer.

    `NonAllocatingRunner` comes first so its methods win over the base's. There
    is deliberately no `__init__`: the base runs all of its own before a
    subclass body would get control, so there is no point at which state set
    here would be visible to it.
    """

    def _setup_device_and_distributed(self, rank, config):
        start_on_host(self, rank, config)


_UNANSWERED = unanswered_rpc_names(CompassModelRunner)
if _UNANSWERED:
    # Checked here rather than left to a deployment. The worker resolves each
    # RPC with `getattr(runner, name, None)` and skips what comes back None, so
    # a name this class stops answering -- because ATOM renamed or dropped it --
    # produces no error anywhere: the worker stays healthy and nothing logs.
    # Failing the import turns that into a worker that dies at construction,
    # with a traceback.
    #
    # The two halves below fail differently, which is why the message
    # partitions them instead of asserting one story for all twelve. A waited
    # name parks its caller on an unbounded queue read for the life of the
    # process. An unwaited one parks nobody: `busy_loop` skips it and carries
    # on -- which for `exit` means `ModelRunner.exit` never runs: the
    # distributed environment is never destroyed, `self.model` is never
    # dropped, and `torch.cuda.empty_cache()` never runs. Its five KV-tensor
    # deletions are `hasattr`-guarded and find nothing here, because this
    # runner allocated none; a drafter it drops is built on meta. The loop
    # breaks either way, since the break is a sibling of the per-runner loop
    # and tests the dispatched name rather than any reply. Skipping
    # `process_kvconnector_output` means a KV transfer is silently never
    # started. Both are real failures; neither is a park.
    #
    # Two things about this raise itself. The CPU gate cannot execute it:
    # importing this module imports `ModelRunner`, which runs aiter's
    # architecture probe and needs a driver, so a green CPU gate is not
    # evidence that the composed class answers the surface -- only an import on
    # a machine with a GPU is. And when it does fire inside a worker,
    # `AsyncIOProc.__init__` (`atom/model_engine/async_proc.py`) resolves the
    # runner class before assigning `self.runners = []`, so the atexit finalizer
    # then fails on the half-built object and the worker log *ends* with
    # `AttributeError: 'AsyncIOProc' object has no attribute 'runners'`. The
    # refusal is the traceback above that one.
    _WAITED = [name for name in _UNANSWERED if RPC_SURFACE[name]]
    _UNREAD = [name for name in _UNANSWERED if not RPC_SURFACE[name]]
    raise RunnerRefusal(
        "the worker dispatches "
        + ", ".join(_UNANSWERED)
        + " by name and this runner answers none of them. Waited on, so a hole "
        "parks its caller for the life of the process: "
        + (", ".join(_WAITED) or "none")
        + ". Dispatched with no reader, so a hole is skipped and the worker "
        "carries on without it: " + (", ".join(_UNREAD) or "none") + "."
    )


class CompassRapidServeModelRunner(RapidServeRpcs, CompassModelRunner):
    """`CompassModelRunner` in RapidServe's prefill and decode engines, with the
    RPCs their engine cores wait on (`RapidServeRpcs`)."""
