# SPDX-License-Identifier: MIT
"""compass_harness puts aiperf's results-affecting time sources on the traffic LP clock.

aiperf's own phase lifecycle, phase runner deadline and system idle cap run on
a real ``TrafficLP`` whose runtime grants each clock call its target at once,
with the event loop's timers made to raise. Skips by name as
``test_harness_pacing.py`` does.
"""

import asyncio
from importlib import metadata
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _missing() -> str | None:
    try:
        metadata.version("compass-harness")
        import compass_harness  # noqa: F401  checks aiperf against its pinned source
    except metadata.PackageNotFoundError as e:
        return f"{e.name} is not installed"
    except RuntimeError as e:
        return str(e)
    return None


if _why := _missing():
    pytest.skip(
        f"needs agentx-harness 56a0cf70 and compass-harness: {_why}",
        allow_module_level=True,
    )

from aiperf.plugin import plugins  # noqa: F401  plugin discovery, as in aiperf

# isort: split
import aiperf.cli_runner
from aiperf.common.enums import CreditPhase
from aiperf.plugin.enums import TimingMode
from aiperf.timing import phase_orchestrator
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.trajectory_source import TrajectorySource
from compass_harness.scheduler import ClockPacedLoopScheduler
from compass_harness.timesource import EPOCH_NS, ClockTimer
from compass_harness.traffic_lp import TrafficLP

DURATION = 1800.0
CAP = 10.0


class GrantAtOnce:
    """The traffic LP's runtime: each clock call is granted its target."""

    def __init__(self) -> None:
        self.now = 0.0
        self.released = {"frontend->traffic:stream": set()}
        self.calls = []

    def next_event(self, t):
        self.calls.append(t)
        self.now = t
        return t

    def close(self) -> None:
        pass


def _wall_timer(*args, **kwargs):
    raise AssertionError("an event-loop timer was armed")


async def _traffic(tmp_path, monkeypatch) -> TrafficLP:
    traffic = TrafficLP(GrantAtOnce(), str(tmp_path / "lp"))
    monkeypatch.setattr(ClockPacedLoopScheduler, "clock", traffic)
    loop = asyncio.get_running_loop()
    loop.call_later = loop.call_at = _wall_timer
    return traffic


def _runner(*, system_cap=None, **phase):
    """The phase runner the orchestrator builds, for AGENTIC_REPLAY profiling."""
    source = MagicMock(spec=TrajectorySource)
    source.trajectories = []
    run = MagicMock()
    run.cfg.get_default_dataset.return_value = SimpleNamespace(
        trace_idle_gap_cap_seconds=None
    )
    run.cfg.get_profiling_phases.return_value = [
        SimpleNamespace(system_idle_gap_cap_seconds=system_cap)
    ]
    run.cfg.server_metrics_disabled = run.cfg.gpu_telemetry_disabled = True
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.AGENTIC_REPLAY,
        total_expected_requests=1,
        expected_duration_sec=DURATION,
        **phase,
    )
    return phase_orchestrator.PhaseRunner(
        config=config,
        conversation_source=source,
        phase_publisher=AsyncMock(),
        credit_router=MagicMock(),
        concurrency_manager=MagicMock(),
        cancellation_policy=MagicMock(),
        callback_handler=MagicMock(),
        run=run,
    )


def test_the_sending_deadline_ends_the_phase_at_its_simulated_duration(
    tmp_path, monkeypatch
):
    """Nothing is sent: the phase's sending wait times out at 1800 simulated s."""

    async def main():
        traffic = await _traffic(tmp_path, monkeypatch)
        runner = _runner()
        runner._lifecycle.start()
        runner._execution_task = asyncio.ensure_future(asyncio.Event().wait())
        await runner._wait_for_sending_complete(MagicMock())
        traffic.done.cancel()
        return runner._lifecycle, runner._execution_task, traffic

    lifecycle, execution, traffic = asyncio.run(main())
    assert traffic.rt.calls == [DURATION]
    assert lifecycle.timeout_triggered and execution.cancelled()
    # Started at simulated 0, yet a stamp aiperf does not read as missing.
    assert lifecycle.started_at_ns and lifecycle.started_at_ns == EPOCH_NS
    assert lifecycle.started_at_perf_ns == EPOCH_NS
    assert lifecycle.sending_complete_at_ns == EPOCH_NS + round(DURATION * 1e9)
    assert lifecycle.time_left_in_seconds() == 0.0


def test_a_deadline_met_early_withdraws_its_clock_wait(tmp_path, monkeypatch):
    """The event is set at 5 s: the 1800 s deadline asks the clock for nothing."""

    async def main():
        traffic = await _traffic(tmp_path, monkeypatch)
        runner = _runner()
        event = asyncio.Event()

        async def set_at_5():
            await traffic.advance_to(5.0)
            event.set()

        setter = asyncio.ensure_future(set_at_5())
        timed_out = await runner._wait_for_event_with_timeout(
            name="sending", event=event, timeout=DURATION, task_to_cancel=None
        )
        await setter
        for _ in range(100):  # let the owner settle with nothing left to ask for
            await asyncio.sleep(0)
        traffic.done.cancel()
        return timed_out, traffic

    timed_out, traffic = asyncio.run(main())
    assert not timed_out
    assert traffic.rt.calls == [5.0] and traffic._waiters == []


def test_a_wait_on_an_event_already_set_leaves_the_clock_alone(tmp_path, monkeypatch):
    """The event is set before the wait: the 1800 s deadline is never armed."""

    async def main():
        traffic = await _traffic(tmp_path, monkeypatch)
        event = asyncio.Event()
        event.set()
        before = traffic.now()
        timed_out = await _runner()._wait_for_event_with_timeout(
            name="sending", event=event, timeout=DURATION, task_to_cancel=None
        )
        for _ in range(100):  # let the owner settle with nothing left to ask for
            await asyncio.sleep(0)
        traffic.done.cancel()
        return timed_out, before, traffic

    timed_out, before, traffic = asyncio.run(main())
    assert not timed_out
    assert (before, traffic.now(), traffic.rt.calls) == (0.0, 0.0, [])


def test_a_deadline_waits_for_a_return_still_inside_aiperfs_callback(
    tmp_path, monkeypatch
):
    """The callback awaits wall-clock I/O while a 100 s deadline is pending:
    every task is blocked, yet the clock must not move until it returns."""

    async def main():
        traffic = TrafficLP(GrantAtOnce(), str(tmp_path / "lp"))
        monkeypatch.setattr(ClockPacedLoopScheduler, "clock", traffic)
        seen, fired = [], asyncio.get_running_loop().create_future()

        async def on_return(worker_id, ret):
            await asyncio.sleep(0.05)
            seen.append(traffic.now())

        ClockTimer(100.0, lambda: fired.set_result(traffic.now()))
        credit = SimpleNamespace(id=0, phase=CreditPhase.PROFILING, phase_index=0)
        await traffic.hold(on_return)("w", SimpleNamespace(credit=credit, error="x"))
        at = await fired
        traffic.done.cancel()
        return seen, at

    assert asyncio.run(main()) == ([0.0], 100.0)


def test_an_idle_gap_over_the_cap_replays_as_the_cap_on_the_clock(
    tmp_path, monkeypatch
):
    """A turn due 100 s after the last return, with the system idle cap at 10 s.

    A scheduler task is still running at the return, so upstream defers the
    cap to its watchdog; the watchdog measures idle on the clock and pulls the
    turn in to the cap.
    """

    async def main():
        traffic = await _traffic(tmp_path, monkeypatch)
        runner = _runner(system_cap=CAP)
        strategy = runner._build_strategy()
        scheduler = runner._scheduler
        started = asyncio.get_running_loop().create_future()

        async def turn():
            started.set_result(traffic.now())

        busy = asyncio.Event()
        scheduler.schedule_later(100.0, turn())
        scheduler.execute_async(busy.wait())
        strategy._progress = SimpleNamespace(in_flight=0)
        strategy.enforce_system_idle_cap(0)  # the return
        at = await started
        busy.set()
        traffic.done.cancel()
        return at, strategy

    at, strategy = asyncio.run(main())
    assert at == CAP
    assert strategy._system_idle_seconds_skipped == 100.0 - CAP


def test_the_accelerated_warmup_is_refused_by_name(tmp_path, monkeypatch):
    async def main():
        await _traffic(tmp_path, monkeypatch)
        _runner(agentic_cache_warmup_duration_sec=30.0)._build_strategy()

    with pytest.raises(
        ValueError,
        match=r"turn them off: accelerated warmup \(--agentic-cache-warmup-duration\)",
    ):
        asyncio.run(main())


def test_the_benchmark_id_repeats_from_run_to_run():
    assert aiperf.cli_runner.uuid4().hex[:12] == "000000000000"
