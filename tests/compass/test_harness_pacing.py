# SPDX-License-Identifier: MIT
"""compass_harness paces aiperf's agentic replay on the Compass clock.

Each pacing call on aiperf's AGENTIC_REPLAY path is driven through the upstream
method that makes it, on the objects a real ``PhaseRunner`` built, with a
stand-in clock and with the event loop's timers made to raise. Needs
compass-harness installed in the interpreter that runs this file, beside an
aiperf whose pinned functions match (agentx-harness 56a0cf70); skips by name
otherwise.
"""

import asyncio
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path
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

# Plugin discovery first, as in aiperf's own processes: it installs the hook
# that rebinds the runner's scheduler.
from aiperf.plugin import plugins

# isort: split
import aiperf
import aiperf.timing.phase.runner as runner_module
from aiperf.common.enums import CreditPhase
from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.request_cancellation import RequestCancellationConfig
from aiperf.timing.trajectory_source import TrajectorySource
from compass_harness.scheduler import ClockPacedLoopScheduler
from compass_harness.strategy import CompassAgenticReplay

AIPERF = Path(aiperf.__file__).parent
T0 = 1000.0
PACING = (
    "schedule_later|schedule_at|schedule_at_perf_sec|schedule_at_perf_ns"
    "|cap_pending_delay|cap_pending_delay_for_group|execute_async"
)
OTHER_TIMING_MODES = {
    "common/loop_scheduler.py",  # its docstring example
    "timing/strategies/fixed_schedule.py",
    "timing/strategies/request_rate.py",
    "timing/strategies/user_centric_rate.py",
}


class StandInClock:
    def __init__(self) -> None:
        self.t = T0

    def now(self) -> float:
        return self.t

    async def advance_to(self, t: float) -> None:
        self.t = t


def _wall_timer(*args, **kwargs):
    raise AssertionError("an event-loop timer was armed")


async def _nothing() -> None:
    pass


def _build(
    clock,
    monkeypatch,
    *,
    seamless=False,
    trace_cap=None,
    system_cap=None,
    metrics_off=True,
    gpu_off=True,
    **phase,
):
    """A real PhaseRunner for AGENTIC_REPLAY profiling, its collaborators mocked."""
    monkeypatch.setattr(ClockPacedLoopScheduler, "clock", clock)
    source = MagicMock(spec=TrajectorySource)
    source.trajectories = []
    run = MagicMock()
    run.cfg.get_default_dataset.return_value = SimpleNamespace(
        trace_idle_gap_cap_seconds=trace_cap
    )
    run.cfg.get_profiling_phases.return_value = [
        SimpleNamespace(system_idle_gap_cap_seconds=system_cap)
    ]
    run.cfg.server_metrics_disabled = metrics_off
    run.cfg.gpu_telemetry_disabled = gpu_off
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.AGENTIC_REPLAY,
        total_expected_requests=1,
        seamless=seamless,
        **phase,
    )
    return runner_module.PhaseRunner(
        config=config,
        conversation_source=source,
        phase_publisher=MagicMock(),
        credit_router=MagicMock(),
        concurrency_manager=MagicMock(),
        cancellation_policy=MagicMock(),
        callback_handler=MagicMock(),
        run=run,
    )


def _record(scheduler, clock) -> dict:
    """Map each aiperf file:line that paced work to the clock time the work started.

    A timer a cap moved earlier is credited to the cap's call site.
    """
    fired, owner = {}, {}

    def caller():
        frame = sys._getframe(2)
        path = Path(frame.f_code.co_filename)
        return (
            (path.relative_to(AIPERF).as_posix(), frame.f_lineno)
            if path.is_relative_to(AIPERF)
            else None
        )

    def schedule_later(delay, coro, *, group_id=None):
        site = [caller()]

        async def start():
            fired[site[0]] = clock.now()
            coro.close()

        work = start()
        owner[id(work)] = site
        return ClockPacedLoopScheduler.schedule_later(
            scheduler, delay, work, group_id=group_id
        )

    def capping(name):
        def cap(*args):
            site = caller()
            before = {id(coro): timer.at for timer, coro in scheduler._handles.values()}
            shifted = getattr(ClockPacedLoopScheduler, name)(scheduler, *args)
            for timer, coro in scheduler._handles.values():
                if timer.at != before[id(coro)]:
                    owner[id(coro)][0] = site
            return shifted

        return cap

    scheduler.schedule_later = schedule_later
    scheduler.cap_pending_delay = capping("cap_pending_delay")
    scheduler.cap_pending_delay_for_group = capping("cap_pending_delay_for_group")
    return fired


async def _settle(scheduler) -> None:
    for _ in range(1000):
        if not scheduler._handles and not scheduler._tasks:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"scheduler did not drain: {scheduler!r}")


# Each drives one upstream method to its pacing call and returns the delay, in
# seconds from T0, at which the paced work should start.


async def _tree_drained(s, o, b, sched):
    s._correlation_to_lane["root"] = 0
    s._on_tree_drained("root", CreditPhase.PROFILING)
    return 0.0


async def _system_idle_cap(s, o, b, sched):
    sched.schedule_later(100.0, _nothing())
    s._system_idle_gap_cap_seconds = 5.0
    s._arm_system_idle_watchdog = (
        lambda delay: None
    )  # a clock timer: test_harness_timesource.py
    s.enforce_system_idle_cap(0)
    return 5.0


async def _warmup_spread(s, o, b, sched):
    states = [
        SimpleNamespace(
            warmup_turn_index=0,
            conversation_id=c,
            x_correlation_id=c,
            root_correlation_id="root",
        )
        for c in ("early", "late")
    ]
    cs = s.conversation_source
    cs.trajectories = [
        SimpleNamespace(
            snapshot=SimpleNamespace(t_star_ms=10_000.0, states=states),
            conversation_id="early",
        )
    ]
    cs.session_for_state = lambda state: MagicMock(
        x_correlation_id=state.x_correlation_id
    )
    stamps = {"early": 0.0, "late": 6_000.0}
    cs.get_metadata = lambda cid: SimpleNamespace(
        turns=[SimpleNamespace(timestamp_ms=stamps[cid])]
    )
    s._build_turn_for_session = lambda session, index: MagicMock()
    await s._execute_warmup()
    return 6.0


async def _accelerated_warmup(s, o, b, sched):
    s._cache_warmup_duration = 3.0
    await s._start_accelerated_warmup()
    return 3.0


async def _next_turn(s, o, b, sched):
    s.conversation_source.get_next_turn_metadata = lambda credit: SimpleNamespace(
        delay_ms=2_500.0, has_forks=False
    )
    credit = SimpleNamespace(
        conversation_id="c", x_correlation_id="c", turn_index=0, num_turns=3, agent_depth=0,
        parent_correlation_id=None, root_correlation_id=None, counts_toward_phase_target=True,
        branch_mode=None, cache_bust_marker=None, cache_bust_target=None, effective_root_correlation_id="c",
    )  # fmt: skip
    await s._dispatch_next_turn(credit)
    return 2.5


async def _profiling_snapshot(s, o, b, sched):
    state = SimpleNamespace(
        x_correlation_id="c", root_correlation_id="c", conversation_id="c", waiting_on_children=False,
        next_dispatch_offset_ms=7_000.0, agent_depth=1, next_turn_index=0,
    )  # fmt: skip
    s._get_snapshot = lambda trajectory: SimpleNamespace(states=[state])
    s._build_turn_for_session = lambda session, index: MagicMock(
        effective_root_correlation_id="c"
    )
    s._lane_root_corr = lambda snapshot: None
    s.branch_orchestrator = None
    await s._dispatch_snapshot_for_profiling(
        SimpleNamespace(conversation_id="c"), 0, 0.0
    )
    return 7.0


async def _child_offset(s, o, b, sched):
    o._start_delayed_first_turn(
        SimpleNamespace(effective_root_correlation_id="child"), 1_500.0, "parent"
    )
    return 1.5


async def _join_deadline(s, o, b, sched):
    pending = SimpleNamespace(
        replay_deadline_armed=False, parent_x_correlation_id="parent", gated_turn_index=2,
        parent_root_correlation_id="root",
    )  # fmt: skip
    o._arm_join_replay_deadline(pending, 2_000.0)
    return 2.0


async def _root_idle_cap(s, o, b, sched):
    sched.schedule_later(100.0, _nothing(), group_id="root")
    b._root_state("root")
    b._enforce_root_idle_cap("root")
    return 0.0


DRIVE = {
    ("timing/strategies/agentic_replay.py", 390): _tree_drained,
    ("timing/strategies/agentic_replay.py", 552): _system_idle_cap,
    ("timing/strategies/agentic_replay.py", 763): _warmup_spread,
    ("timing/strategies/agentic_replay.py", 810): _accelerated_warmup,
    ("timing/strategies/agentic_replay.py", 1560): _next_turn,
    ("timing/strategies/agentic_replay.py", 1797): _profiling_snapshot,
    ("timing/branch_orchestrator.py", 1265): _child_offset,
    ("timing/branch_orchestrator.py", 1467): _join_deadline,
    ("timing/replay_dependencies.py", 319): _root_idle_cap,
}


def test_the_driven_sites_are_every_pacing_call_on_the_agentic_path():
    found = {
        (rel, n)
        for path in AIPERF.rglob("*.py")
        if (rel := path.relative_to(AIPERF).as_posix()) not in OTHER_TIMING_MODES
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if re.search(rf"scheduler\.({PACING})\(", line)
    }
    assert found == set(DRIVE)


@pytest.mark.parametrize(
    "site", sorted(DRIVE), ids=lambda site: f"{Path(site[0]).name}:{site[1]}"
)
def test_each_pacing_call_advances_on_the_clock_and_never_the_wall(site, monkeypatch):
    clock = StandInClock()

    async def main():
        loop = asyncio.get_running_loop()
        runner = _build(clock, monkeypatch)
        strategy = runner._build_strategy()
        strategy.credit_issuer = MagicMock(issue_credit=AsyncMock())
        fired = _record(runner._scheduler, clock)
        loop.call_later = loop.call_at = _wall_timer
        delay = await DRIVE[site](
            strategy,
            runner._branch_orchestrator,
            runner._replay_barrier,
            runner._scheduler,
        )
        await _settle(runner._scheduler)
        return fired, delay

    fired, delay = asyncio.run(main())
    print(f"\n{site[0]}:{site[1]} started at clock {fired.get(site)}")
    assert fired == {site: T0 + delay}


def test_discovery_rebinds_the_runner_scheduler_for_all_three_holders(monkeypatch):
    async def main():
        runner = _build(StandInClock(), monkeypatch)
        return runner, runner._build_strategy()

    runner, strategy = asyncio.run(main())
    assert (
        plugins.get_class("timing_strategy", "agentic_replay") is CompassAgenticReplay
    )
    assert type(runner._scheduler) is ClockPacedLoopScheduler
    assert strategy.scheduler is runner._scheduler
    assert runner._branch_orchestrator._scheduler is runner._scheduler
    assert runner._replay_barrier._scheduler is runner._scheduler


def test_without_the_rebind_the_strategy_refuses_naming_the_scheduler(monkeypatch):
    monkeypatch.setattr(runner_module, "LoopScheduler", LoopScheduler)

    async def main():
        _build(StandInClock(), monkeypatch)._build_strategy()

    with pytest.raises(TypeError, match="scheduler is LoopScheduler"):
        asyncio.run(main())


def test_an_unregistered_plugin_fails_the_registration_check():
    script = (
        "import importlib.metadata as m\n"
        "found = m.entry_points\n"
        "m.entry_points = lambda **kw: [ep for ep in found(**kw) if ep.name != 'compass']\n"
        "import compass_harness\n"
        "import aiperf.timing.phase.runner\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode != 0
    assert (
        "Compass plugin not registered: timing_strategy agentic_replay is aiperf."
        in done.stderr
    )


def test_an_unbound_clock_refuses_the_phase_runner(monkeypatch):
    async def main():
        _build(None, monkeypatch)

    with pytest.raises(RuntimeError, match="Compass clock not bound"):
        asyncio.run(main())


class GrantingClock:
    """A clock whose ``advance_to`` suspends until ``grant()`` wakes every wait at once."""

    def __init__(self) -> None:
        self.t = T0
        self.waits: list[tuple[float, asyncio.Future]] = []

    def now(self) -> float:
        return self.t

    async def advance_to(self, t: float) -> None:
        fut = asyncio.get_running_loop().create_future()
        self.waits.append((t, fut))
        await fut

    def grant(self) -> None:
        waits, self.waits = self.waits, []
        self.t = min(t for t, _ in waits)
        for _, fut in waits:
            fut.set_result(None)


def test_a_phase_end_cancel_with_a_timer_pending_leaves_the_clock_wait_whole(
    monkeypatch,
):
    """The runner's phase-end ``cancel_all()`` while the driver waits on the clock."""

    async def main():
        clock = GrantingClock()
        monkeypatch.setattr(ClockPacedLoopScheduler, "clock", clock)
        scheduler = ClockPacedLoopScheduler()
        started = []

        async def turn(name):
            started.append(name)

        scheduler.schedule_later(5.0, turn("cancelled"))
        await asyncio.sleep(0)
        assert [t for t, _ in clock.waits] == [T0 + 5.0]
        scheduler.cancel_all()
        scheduler.schedule_later(1.0, turn("armed after the cancel"))
        for _ in range(10):
            await asyncio.sleep(0)
            if clock.waits:
                clock.grant()
        return started, scheduler.pending_count

    assert asyncio.run(main()) == (["armed after the cancel"], 0)


@pytest.mark.parametrize(
    "earlier, expected",
    [
        (
            lambda scheduler, turn: scheduler.schedule_later(1.0, turn("due +1")),
            [("due +1", 1.0), ("due +5", 5.0)],
        ),
        (
            lambda scheduler, turn: scheduler.cap_pending_delay(1.0),
            [("due +5", 1.0)],
        ),
    ],
    ids=["armed", "capped"],
)
def test_a_timer_due_before_the_driver_wait_starts_at_its_own_time(
    earlier, expected, monkeypatch
):
    """A timer armed, or capped, earlier than the time the driver already waits for."""

    async def main():
        clock = GrantingClock()
        monkeypatch.setattr(ClockPacedLoopScheduler, "clock", clock)
        scheduler = ClockPacedLoopScheduler()
        started = []

        async def turn(name):
            started.append((name, clock.now() - T0))

        scheduler.schedule_later(5.0, turn("due +5"))
        await asyncio.sleep(0)
        assert [t for t, _ in clock.waits] == [T0 + 5.0]
        earlier(scheduler, turn)
        for _ in range(20):
            await asyncio.sleep(0)
            if clock.waits:
                clock.grant()
        return started

    assert asyncio.run(main()) == expected


@pytest.mark.parametrize(
    "option, name",
    [
        ({"seamless": True}, "seamless"),
        ({"trace_cap": 30.0}, "trace_idle_gap_cap_seconds"),
        ({"metrics_off": False}, "server metrics"),
        ({"gpu_off": False}, "GPU telemetry"),
        ({"prefill_concurrency": 2}, "prefill concurrency (--prefill-concurrency)"),
        (
            {"request_cancellation": RequestCancellationConfig(rate=10.0)},
            "request cancellation (--request-cancellation-rate)",
        ),
    ],
)
def test_each_unpaced_option_is_refused_by_name(option, name, monkeypatch):
    async def main():
        _build(StandInClock(), monkeypatch, **option)._build_strategy()

    with pytest.raises(ValueError, match=f"turn them off: {re.escape(name)}"):
        asyncio.run(main())
