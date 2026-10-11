# SPDX-License-Identifier: MIT
"""A clock call refused on an engine-side thread reaches the run summary.

`engine_done` writes the engine process's refusals to the run's out_dir, and
`frontend_done` folds them into ``summary.json`` beside the frontend's own.
"""

import json
import math
import threading
import time
from types import SimpleNamespace

import pytest

from atom.compass import run as compass_run
from atom.utils import clock
from atom.utils.clock import LPRuntime


def test_a_clock_call_refused_on_an_engine_thread_is_a_summary_refusal(
    monkeypatch, tmp_path
):
    run = {
        "admission_path": "serving",
        "ipc_s": 0.001,
        "stream_s": 0.002,
        "bound_s": 10.0,
        "clock_endpoint": "inproc:unused",
        "out_dir": str(tmp_path),
    }
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    table = compass_run.channel_table(run)
    authority = compass_run._RecordingAuthority(run)
    authority.started = authority.finished = time.monotonic()

    # The engine's runtime after the +inf grant; a thread that is not its clock
    # owner asks to advance, and the caller drops the exception.
    engine = LPRuntime(compass_run.ENGINE, table, None)
    engine.now = math.inf
    monkeypatch.setattr(clock, "_installed", engine)

    def call():
        with pytest.raises(RuntimeError, match="only the clock owner"):
            engine.advance_to(1.0)

    t = threading.Thread(target=call, name="EngineOutputThread")
    t.start()
    t.join()
    worker = SimpleNamespace(call_func=lambda name, wait_out: ())
    compass_run.engine_done(SimpleNamespace(runner_mgr=worker))

    # The frontend's side: no refusals of its own.
    monkeypatch.setattr(compass_run, "_authority", authority)
    frontend = LPRuntime(compass_run.FRONTEND, table, None)
    frontend.now = math.inf
    frontend.loop = SimpleNamespace(executor=SimpleNamespace(refusals=[]))
    monkeypatch.setattr(clock, "_installed", frontend)
    assert compass_run.frontend_done(SimpleNamespace(close=lambda: None))

    summary = json.loads((tmp_path / compass_run.SUMMARY_FILE).read_text())
    refusals = summary["schedule"]["refusals"]
    print("refusals:", json.dumps(refusals))
    assert refusals["reasons"] == [["clock:advance_to from EngineOutputThread", 1]]
