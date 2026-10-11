# SPDX-License-Identifier: MIT
"""A run whose engine raises mid-run still writes its summary.

ATOM's `EngineCore.run_engine` runs an engine whose loop takes one step and
then raises, before the finish grant: an exception, or the `SystemExit` a dead
worker raises; or it leaves the loop without raising, as a shutdown request
ends it, and the clock refuses to close. The authority's host, co-hosted in the
frontend or standalone, then writes ``summary.json`` with each LP's clock as
it stands, the exception named, and the run marked a coverage report.
"""

import json
import threading
import time

import pytest
from aiter_stub import stubbed_aiter
from test_dp_slice import _deployment

from atom.compass import run as compass_run
from atom.compass.clock_transport import service
from atom.utils import clock
from atom.utils.distributed import utils


@pytest.mark.parametrize(
    "error, reason",
    [
        (RuntimeError("the step raised"), "RuntimeError('the step raised')"),
        # How `call_func` reports a dead worker.
        (SystemExit(), "SystemExit()"),
        # A shutdown request mid-run: the loop returns before the finish.
        (
            None,
            (
                "RuntimeError('engine left its loop at 0.0005, before the +inf "
                "grant finished the run')"
            ),
        ),
    ],
    ids=["exception", "dead worker", "left early"],
)
@pytest.mark.parametrize("host", ["co-hosted", "standalone"])
def test_an_engine_that_raises_mid_run_leaves_a_summary_naming_it(
    monkeypatch, tmp_path, host, error, reason
):
    run = {
        "admission_path": "serving",
        "ipc_s": 0.001,
        "stream_s": 0.002,
        "bound_s": 10.0,
        "clock_endpoint": f"inproc:{tmp_path}",
        "out_dir": str(tmp_path),
    }
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    for module, name in [
        (compass_run, "_authority"),
        (compass_run, "_member"),
        (clock, "_installed"),
        (utils, "LP_OF_RANK"),
    ]:
        monkeypatch.setattr(module, name, None)
    if host == "co-hosted":
        with compass_run.frontend(_deployment()):
            pass
    else:
        threading.Thread(
            target=compass_run.authority, args=(run["clock_endpoint"],), daemon=True
        ).start()
        deadline = time.monotonic() + 10
        while run["clock_endpoint"] not in service._SERVED:
            assert time.monotonic() < deadline, "no authority in 10 seconds"
            time.sleep(0.01)

    with stubbed_aiter():
        from atom.model_engine import engine_core

    class RaisingEngine:
        """One step, then the loop raises or returns. It has no workers to ask."""

        _setup_engine_process = staticmethod(lambda name: None)

        def __init__(self, *args) -> None:
            pass

        def busy_loop(self) -> None:
            # Below the frontend's lookahead, so granted while it runs at 0.
            clock.installed().advance_to(run["ipc_s"] / 2)
            if error is not None:
                raise error

        def exit(self) -> None:
            pass

    run_engine = engine_core.EngineCore.run_engine
    monkeypatch.setattr(engine_core, "EngineCore", RaisingEngine)
    with pytest.raises(RuntimeError if error is None else type(error)):
        run_engine(_deployment(), "", "")

    path = tmp_path / compass_run.SUMMARY_FILE
    deadline = time.monotonic() + 10
    while not path.exists():
        assert time.monotonic() < deadline, "no summary.json in 10 wall seconds"
        time.sleep(0.05)
    summary = json.loads(path.read_text())
    schedule = summary["schedule"]
    print("summary:", json.dumps(schedule))
    assert schedule["refusals"]["reasons"] == [[f"ended:engine raised {reason}", 1]]
    assert dict(schedule["final_clocks"])["engine"] == run["ipc_s"] / 2
    assert summary["coverage_report"] is True
    assert (tmp_path / compass_run.STEP_TABLE_FILE).exists()
