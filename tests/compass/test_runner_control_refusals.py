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
from atom.model_engine.collective_rpc import DISCARD_WEIGHT_SYNC_CMD, RpcResult
from atom.model_engine.engine_utility import EngineUtilityHandler
from atom.model_engine.sequence import SequenceStatus

# The calls a handler sends a worker method through, and where the name is:
# `call_func` takes it first; the all-rank helpers take the command, then it.
NAME_ARG = {"call_func": 0, "_update_on_every_rank": 1, "_utility_on_every_rank": 1}


def _worker_method(cmd):
    """The name a command's handler broadcasts to the worker, or None."""
    handler = getattr(EngineUtilityHandler, EngineUtilityHandler._UTILITY_HANDLERS[cmd])
    tree = ast.parse(textwrap.dedent(inspect.getsource(handler)))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    names = [
        n.args[NAME_ARG[n.func.attr]].value
        for n in calls
        if getattr(n.func, "attr", None) in NAME_ARG
    ]
    # The discard helper sends its one method by constant rather than literal.
    names += [
        DISCARD_WEIGHT_SYNC_CMD
        for n in calls
        if getattr(n.func, "attr", None) == "_discard_failed_update_on_every_rank"
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

    def utility_rpc(self, name, payload, timeout):
        """`AsyncIOProc._run_generic_rpc` for one rank: every outcome is a reply."""
        func = getattr(self.runner, name, None)
        if func is None:
            return [RpcResult(payload.request_id, 0, error=f"no method {name!r}")]
        out = func(*payload.args)
        self.calls.append((name, out))
        return [RpcResult(payload.request_id, 0, value=out)]


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
    # `collective_rpc` names its method in the request, so no one method is read.
    assert {cmd for cmd, method in REACHES.items() if not method} == {
        "abort_request",
        "get_mtp_stats",
        "get_mtp_statistics",
        "get_cache_statistics",
        "finish_weight_sync",
        "get_weight_sync_status",
        "collective_rpc",
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
    # The handler sends the refusal back as its result, where it sends one;
    # the discard handler answers its own True, whatever the workers said.
    assert all(
        r == ("UTILITY_RESPONSE", {"cmd": cmd, "result": reason})
        for r in responses[:-1]
        if cmd != DISCARD_WEIGHT_SYNC_CMD
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
        {
            "cmd": "finish_weight_sync",
            "error": "weight sync None is not the sync this engine is in (None)",
        },
        {
            "cmd": "get_weight_sync_status",
            "result": {"fenced": False, "in_progress": False, "failure": None},
        },
        {"cmd": "get_mtp_statistics", "result": {"enabled": False}},
        {"cmd": "get_cache_statistics", "result": {"enabled": False}},
        {
            "cmd": "collective_rpc",
            "request_id": None,
            "error": "collective_rpc needs a method name and a request id, both "
            "non-empty strings",
        },
    ]
    # Nothing refused is still an answer, so the reader is never parked.
    assert Worker(runner).call_func("refused_commands", wait_out=True) == ()
