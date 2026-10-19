# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The model runner a simulated run installs in place of ATOM's own.

The modules here, split by what each is allowed to import:

- `overrides` holds the behaviour -- the methods that keep construction off the
  device -- and imports nothing from the engine at module scope. That is not a
  stylistic preference. Importing `atom.model_engine.model_runner` runs aiter's
  architecture probe, which shells out to `rocminfo` and raises in a container
  with no driver, so anything reachable from that import cannot be exercised by
  a test that runs without one. The engine names it needs are imported where
  they are used -- `ScheduledBatchOutput` inside `forward`, and
  `DeviceMemoryReadings` inside `_read_device_memory` -- and both of those only
  ever run on a worker that has imported the engine already.
- `step_output` holds what a predicted step reports -- the deferral rules and
  the token ids -- over numpy and nothing else, so it has no engine import to
  defer.
- `model_runner` binds those overrides onto `ModelRunner`, and is therefore the
  only module here that needs a driver to import. It is also where the composed
  class is checked against `overrides.RPC_SURFACE`, the table of names a worker
  dispatches.

A hole in that surface is quiet either way: the worker skips a name the runner
lacks and carries on without raising. It is not one failure, though, and
`RPC_SURFACE`'s own column is what says which one it is. For a name whose
caller waits, a hole parks that caller for the life of the process. For a name
the table marks unwaited, a hole parks nobody, and what is lost is the work the
name stood for:

- `exit` (`EngineCore.exit`) never reaches `ModelRunner.exit`; the comment
  on the `RPC_SURFACE` check in `model_runner` says what that loses here.
  The worker still leaves its loop -- `busy_loop` breaks
  on the dispatched name, in a statement beside the per-runner loop rather than
  inside it -- so the symptom is what shutdown failed to release, not a hang.
- `process_kvconnector_output` never starts the asynchronous KV transfer its
  metadata was built for (a consumer's load, a producer's send, an offload
  save). It is broadcast from `EngineCore._process_engine_step_inner`,
  `EngineCore._dispatch_idle_offload_work`, `PPEngineCoreProc._pp_head_step`,
  `PPEngineCoreProc._dispatch_connector_only_batch` and
  `PPEngineCoreProc._downstream_busy_loop`, and waited for at none of them.

`overrides` states the reply contract both of these follow from, including why
a method that is present and answers None is the same event to a caller as one
that is absent.

This package stays empty at import time for the same reason the split above
exists, so naming it costs nothing.

`COMPASS_RUNNER_QUALNAME` is the string that selects the runner: a `Config`
field, read where the worker process instantiates its runner. Installing this
class needs no change to ATOM.
"""

COMPASS_RUNNER_QUALNAME = "atom.compass.runner.model_runner.CompassModelRunner"
#: Its RapidServe form, which a simulated RapidServe run names.
COMPASS_RAPIDSERVE_QUALNAME = (
    "atom.compass.runner.model_runner.CompassRapidServeModelRunner"
)

__all__ = ["COMPASS_RAPIDSERVE_QUALNAME", "COMPASS_RUNNER_QUALNAME"]
