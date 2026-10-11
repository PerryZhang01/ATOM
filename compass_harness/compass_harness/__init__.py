# SPDX-License-Identifier: MIT
"""Compass adapter for agentx-harness (aiperf).

aiperf's plugin discovery imports this package while ``aiperf.plugin.plugins``
is still initialising, so nothing here imports aiperf at module level: an
import error escaping from here makes discovery drop the whole plugin with one
WARNING line.

Importing the package installs an import hook that rebinds module globals of
each aiperf module in ``REBIND`` once that module has executed. The runner's
``LoopScheduler`` becomes ``ClockPacedLoopScheduler``, so every phase runner's
scheduler, and the branch orchestrator and replay barrier it is handed to, pace
on the Compass clock; the hook then refuses the run unless this package's
strategy is the registered ``agentic_replay``. The timing manager's
``StickyCreditRouter`` becomes ``CompassCreditRouter``, which binds that clock
to the traffic LP. The orchestrator's ``PhaseRunner`` becomes
``ClockPhaseRunner``, whose phase deadlines wait on that clock. ``time`` in the
phase lifecycle, the credit issuer and the strategy reads the clock, so phase
stamps and windows, ``issued_at_ns`` and the system idle cap are simulated; the
CLI's ``uuid4`` is fixed, so prompt token counts repeat from run to run.
"""

import importlib
import importlib.util
import os
import sys

from compass_harness import fingerprint

RUNNER = "aiperf.timing.phase.runner"
STRATEGY = "compass_harness.strategy:CompassAgenticReplay"
#: The path prefix of the adapter's two sockets, shared by every aiperf process.
ADDRESS_ENV = "COMPASS_HARNESS_IPC"
SIM_TIME = (("time", "compass_harness.timesource:SimTime"),)
#: aiperf module -> ((a global to rebind, the replacement as module:qualname), ...).
REBIND = {
    RUNNER: (("LoopScheduler", "compass_harness.scheduler:ClockPacedLoopScheduler"),),
    "aiperf.timing.manager": (
        ("StickyCreditRouter", "compass_harness.router:CompassCreditRouter"),
    ),
    "aiperf.timing.phase_orchestrator": (
        ("PhaseRunner", "compass_harness.strategy:ClockPhaseRunner"),
    ),
    "aiperf.timing.phase.lifecycle": SIM_TIME,
    "aiperf.credit.issuer": SIM_TIME,
    "aiperf.timing.strategies.agentic_replay": SIM_TIME,
    "aiperf.cli_runner": (("uuid4", "compass_harness.timesource:fixed_uuid4"),),
}


def check_registered() -> None:
    """Raise unless ``agentic_replay`` resolves to this package's strategy."""
    from aiperf.plugin import plugins

    found = plugins.get_entry("timing_strategy", "agentic_replay").class_path
    if found != STRATEGY:
        raise RuntimeError(
            f"Compass plugin not registered: timing_strategy agentic_replay is "
            f"{found}, not {STRATEGY}. Look for a 'Plugin discovery' warning "
            "naming compass."
        )


class _RebindHook:
    """Meta-path finder that wraps each ``REBIND`` module's loader, then removes itself."""

    def __init__(self, pending: dict) -> None:
        self.pending = pending

    def find_spec(self, name, path=None, target=None):
        if name not in self.pending:
            return None
        rebinds = self.pending.pop(name)
        if not self.pending:
            sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        execute = spec.loader.exec_module

        def exec_module(module):
            execute(module)
            for attr, replacement in rebinds:
                path, _, qualname = replacement.partition(":")
                setattr(module, attr, getattr(importlib.import_module(path), qualname))
            if name == RUNNER:
                check_registered()

        spec.loader.exec_module = exec_module
        return spec


# Every aiperf process is spawned after the CLI process ran this, so all of
# them share one adapter socket address.
os.environ.setdefault(ADDRESS_ENV, f"/tmp/compass-harness-{os.getpid()}")
if pending := {m: r for m, r in REBIND.items() if m not in sys.modules}:
    sys.meta_path.insert(0, _RebindHook(pending))

# After the hook: when discovery drops a refused plugin, the hook re-imports this
# package as the runner loads, and the run fails on the same message.
fingerprint.check()
