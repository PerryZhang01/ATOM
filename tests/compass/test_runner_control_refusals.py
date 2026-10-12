# SPDX-License-Identifier: MIT
"""ATOM's utility commands, sent through its own handler to the simulated runner.

`EngineUtilityHandler` is ATOM's, unchanged. The worker side is `busy_loop`'s
dispatch for one runner: `getattr(runner, name, None)`, call it, and forward
the reply only when it is not None. `test_runner_rpc_surface.py` pins that
structure over ATOM's source, so a reply `Worker` reports as never forwarded
is one the real worker never forwards either, and its caller waits forever.

Which worker method a command reaches is read off the handler's source, not
listed here.
"""

import ast
import inspect
import queue
import textwrap
from types import SimpleNamespace

import pytest

from atom.compass.clock import RefusalTally
from atom.compass.runner.overrides import NonAllocatingRunner
from atom.model_engine.engine_utility import EngineUtilityHandler
from atom.model_engine.sequence import SequenceStatus


def _worker_method(cmd):
    """The name a command's handler broadcasts to the worker, or None."""
    handler = getattr(EngineUtilityHandler, EngineUtilityHandler._UTILITY_HANDLERS[cmd])
    tree = ast.parse(textwrap.dedent(inspect.getsource(handler)))
    names = [
        n.args[0].value
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "call_func"
    ]
    assert len(names) <= 1, f"{cmd} broadcasts {names}"
    return names[0] if names else None


REACHES = {cmd: _worker_method(cmd) for cmd in EngineUtilityHandler._UTILITY_HANDLERS}
REFUSED = {cmd: method for cmd, method in REACHES.items() if method}


class Worker:
    """`call_func(..., wait_out=True)` over `busy_loop`'s dispatch, one runner."""

    def __init__(self, runner):
        self.runner = runner
        self.calls = []

    def call_func(self, name, *args, wait_out=False):
        func = getattr(self.runner, name, None)
        out = None if func is None else func(*args)
        if out is None:
            raise AssertionError(f"{name} forwards nothing; its caller waits forever")
        self.calls.append((name, out))
        return out


class ModelRunnerProfiler:
    """Stands in for `ModelRunner`'s own profiler pair, which really profiles."""

    def start_profiler(self):
        return "profiling"

    def stop_profiler(self):
        return "profiled"


class Runner(NonAllocatingRunner, ModelRunnerProfiler):
    """Composed in `CompassModelRunner`'s order: the overrides, then ATOM's."""


def _send(runner, *cmds, scheduler=None):
    """Run *cmds* through ATOM's handler; the worker calls and the responses."""
    pending, out = queue.Queue(), queue.Queue()
    for cmd in cmds:
        pending.put((cmd, {"req_id": "r0"}))  # read by abort_request alone
    handler = EngineUtilityHandler(Worker(runner), out, scheduler=scheduler)
    engine = SimpleNamespace(_has_pending_utility=True, _is_rl_weights_offloaded=False)
    handler.process_queue(pending, engine)
    return handler.runner_mgr.calls, [out.get_nowait() for _ in range(out.qsize())]


def test_which_commands_reach_a_worker_is_read_off_atoms_handler():
    assert {cmd for cmd, method in REACHES.items() if not method} == {
        "abort_request",
        "get_mtp_stats",
        "get_mtp_statistics",
        "get_cache_statistics",
    }
    assert {REFUSED["update_weights_shm"], REFUSED["update_weights_ipc"]} == {
        "update_weights_from_shm",
        "update_weights_from_ipc",
    }


@pytest.mark.parametrize("cmd", sorted(REFUSED))
def test_a_worker_command_is_refused_by_name_and_the_engine_keeps_serving(cmd):
    runner = Runner()
    reason = f"command:{REFUSED[cmd]}"
    calls, responses = _send(runner, cmd, "get_mtp_statistics")
    assert calls == [(REFUSED[cmd], reason)]
    assert runner.refused_commands() == (reason,)
    # The handler sends the refusal back as its result, where it sends one.
    assert all(
        r == ("UTILITY_RESPONSE", {"cmd": cmd, "result": reason})
        for r in responses[:-1]
    )
    # And the next command in the queue is still served.
    assert responses[-1] == (
        "UTILITY_RESPONSE",
        {"cmd": "get_mtp_statistics", "result": {"enabled": False}},
    )


def test_the_summary_counts_each_refusal_read_back_over_the_worker_rpc():
    runner = Runner()
    _send(runner, *REACHES)
    reasons = Worker(runner).call_func("refused_commands", wait_out=True)
    record = RefusalTally.of(reasons, steps=0).record()
    assert record["by_source"] == [["command", len(REFUSED)]]
    assert dict(record["reasons"]) == {f"command:{m}": 1 for m in REFUSED.values()}


def test_the_commands_that_reach_no_worker_run_as_in_atom():
    seq = SimpleNamespace(id="r0", status=SequenceStatus.RUNNING)
    scheduler = SimpleNamespace(running=[seq], waiting=[], engine_stats=None)
    runner = Runner()
    calls, responses = _send(
        runner, *(c for c, m in REACHES.items() if not m), scheduler=scheduler
    )
    assert calls == []
    assert seq.status == SequenceStatus.ABORTED
    assert [r[1] for r in responses] == [
        {"cmd": "get_mtp_statistics", "result": {"enabled": False}},
        {"cmd": "get_cache_statistics", "result": {"enabled": False}},
    ]
    # Nothing refused is still an answer, so the reader is never parked.
    assert Worker(runner).call_func("refused_commands", wait_out=True) == ()
