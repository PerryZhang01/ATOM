# SPDX-License-Identifier: MIT
"""A server that serves until it is stopped, stopped with no client run open,
is summarised as the finish would summarise it.

A co-hosting frontend, an engine and one traffic LP per client run run over
the authority the frontend serves (`test_client_run_summary._serve`), and a
SIGTERM stops the server after one client run. A client run is open from the
client's bind to its end. A server whose ``client_runs`` is ``null``, stopped
between client runs, writes a summary with no refusal reason once every
server LP has handed in; one with a client bound, or one that declares how
many client runs it serves, writes an early end naming the signal. Only a
frontend's TAR to ``+inf`` is a stop.
"""

import json
import math
from collections import Counter
from types import SimpleNamespace

import pytest
from test_client_run_summary import _read, _serve

from atom.compass import run as compass_run
from atom.utils import clock
from atom.utils.clock import LPRuntime

INF = math.inf
EARLY = [["ended:frontend stopped by SIGTERM", 1]]
#: One client run that the workers and the executor refuse nothing in.
CLIENT_RUN = ([], {"steps": 3, "seconds": 0.75, "treatments": Counter()}, [])


def _run(monkeypatch, tmp_path, client_runs) -> dict:
    run = {"admission_path": "serving", "ipc_s": 2.0**-10, "stream_s": 2.0**-9}
    run.update(bound_s=100.0, out_dir=str(tmp_path), client_runs=client_runs)
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    return run


@pytest.mark.parametrize(
    "client_runs, stop, reasons",
    [(None, "resigned", []), (None, "bound", EARLY), (2, "resigned", EARLY)],
    ids=["null-between-client-runs", "null-client-bound", "two-between-client-runs"],
)
def test_a_null_server_stopped_with_no_client_run_open_is_summarised_as_finished(
    monkeypatch, tmp_path, client_runs, stop, reasons
):
    out = _serve(monkeypatch, tmp_path / "server", client_runs, [CLIENT_RUN], stop)
    summary = _read(out / compass_run.SUMMARY_FILE)
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
