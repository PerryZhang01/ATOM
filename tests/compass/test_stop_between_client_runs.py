# SPDX-License-Identifier: MIT
"""A server that serves until it is stopped, stopped with no client run open,
is summarised as the finish would summarise it.

A co-hosting frontend, an engine and one traffic LP, each the run bootstrap's
own runtime, run over the authority the frontend serves; the frontend's
runtime stops the run as its event loop does at a signal. A client run is open
from the client's bind to its end. A server whose ``client_runs`` is ``null``,
stopped between client runs, writes a summary with no refusal reason once
every server LP has handed in; one with a client bound, or one that declares
how many client runs it serves, writes an early end naming the signal. Only a
frontend's TAR to ``+inf`` is a stop.
"""

import json
import math
from types import SimpleNamespace

import pytest
from test_dp_slice import _deployment

from atom.compass import run as compass_run
from atom.compass.clock_transport import connect
from atom.utils import clock
from atom.utils.clock import LPRuntime
from atom.utils.distributed import utils

INF = math.inf
HTTP, STREAM = "traffic->frontend:http", "frontend->traffic:stream"
EARLY = [["ended:frontend stopped by SIGTERM", 1]]


def _run(monkeypatch, tmp_path, client_runs) -> dict:
    run = {"admission_path": "serving", "ipc_s": 2.0**-10, "stream_s": 2.0**-9}
    run.update(bound_s=100.0, clock_endpoint=f"inproc:{tmp_path}")
    run.update(out_dir=str(tmp_path), client_runs=client_runs)
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    return run


@pytest.mark.parametrize(
    "client_runs, resigned, reasons",
    [(None, True, []), (None, False, EARLY), (2, True, EARLY)],
    ids=["null-between-client-runs", "null-client-bound", "two-between-client-runs"],
)
def test_a_null_server_stopped_with_no_client_run_open_is_summarised_as_finished(
    monkeypatch, tmp_path, client_runs, resigned, reasons
):
    run = _run(monkeypatch, tmp_path, client_runs)
    for module, name in [
        (compass_run, "_authority"),
        (compass_run, "_early_end"),
        (compass_run, "_member"),
        (clock, "_installed"),
        (utils, "LP_OF_RANK"),
    ]:
        monkeypatch.setattr(module, name, None)
    with compass_run.frontend(_deployment()):
        pass
    frontend = clock.installed()
    frontend.loop = SimpleNamespace(executor=SimpleNamespace(refusals=[]), stopped=None)
    with compass_run.engine(_deployment()):
        pass
    engine = clock.installed()
    said = {"refused_commands": [], "priced_steps": {"steps": 3, "seconds": 0.75}}
    said["predicted_memory"] = {}
    workers = SimpleNamespace(call_func=lambda name, wait_out: said[name])
    with compass_run.engine_loop(SimpleNamespace(runner_mgr=workers)):
        frontend.ask_next_event(INF)
        engine.ask_next_event(INF, 50.0)
        table = compass_run.channel_table(run)
        traffic = LPRuntime(
            compass_run.TRAFFIC,
            table,
            connect(compass_run.TRAFFIC, f"inproc:{tmp_path}"),
        )
        traffic.start_run()
        traffic.stamp_send(HTTP)
        traffic.ask_next_event(INF)
        frontend.take_grant()
        frontend.stamp_send(STREAM)
        frontend.ask_next_event(INF)
        traffic.take_grant()
        if resigned:  # the client run ends: the client resigns, the server stays
            traffic.ask_next_event(INF)
            assert traffic.take_grant() == INF
            assert max(engine.take_grant(), frontend.take_grant()) < INF
            engine.ask_next_event(INF, 50.0)
            frontend.ask_next_event(INF)
        # A SIGTERM, as the frontend's event loop takes it.
        frontend.loop.stopped = "SIGTERM"
        frontend.end_run()
        assert engine.take_grant() == INF
    monkeypatch.setattr(clock, "_installed", frontend)
    assert compass_run.frontend_done(SimpleNamespace(close=lambda: None))
    summary = json.loads((tmp_path / compass_run.SUMMARY_FILE).read_text())
    assert summary["schedule"]["refusals"]["reasons"] == reasons
    assert summary["coverage_report"] is bool(reasons)
    assert summary["schedule"]["predicted_seconds"] == 0.75


def test_only_a_frontends_tar_to_inf_is_a_stop(monkeypatch, tmp_path):
    authority = compass_run._RecordingAuthority(_run(monkeypatch, tmp_path, None))
    with pytest.raises(ValueError, match="the TAR target of engine must be a finite"):
        authority.on_request(compass_run.ENGINE, "TAR", INF, [])
    assert (authority.stopped, authority.final_clocks) == (False, None)


def test_a_whole_stop_waits_for_every_server_lps_hand_in(monkeypatch, tmp_path):
    """As at the finish: the summary is not written without the engine's."""
    run = _run(monkeypatch, tmp_path, None)
    authority = compass_run._RecordingAuthority(run)
    authority.on_request(compass_run.FRONTEND, "TAR", INF, [])
    frontend = LPRuntime(compass_run.FRONTEND, compass_run.channel_table(run), None)
    executor = SimpleNamespace(refusals=[])
    frontend.loop = SimpleNamespace(executor=executor, stopped="SIGTERM")
    monkeypatch.setattr(clock, "_installed", frontend)
    monkeypatch.setattr(compass_run, "_authority", authority)
    monkeypatch.setattr(compass_run, "_early_end", SimpleNamespace(join=lambda: None))
    monkeypatch.setattr(compass_run, "HAND_IN_WAIT_S", 0.2)
    with pytest.raises(TimeoutError, match=r"no lp-engine\.json in"):
        compass_run.frontend_done(SimpleNamespace(close=lambda: None))
