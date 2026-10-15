# SPDX-License-Identifier: MIT
"""Each client run a simulated server serves has its own summary.

A co-hosting frontend, an engine and one traffic LP per client run, each the
run bootstrap's own runtime, run over the authority the frontend serves. At
the end of a client run before the last, the engine and the frontend each
hand in at their marked grant, from inside their clock call, and the
authority's host writes that client run's summary; the finish writes the last
one's beside the run summary. A client run's counts are its own: the engine's
workers and the frontend's executor count from the server's start, and a
summary takes each hand-in less the one at the previous client run's end, so
it matches the summary a fresh server writes for the same client run. A stop
after a client run writes the run summary and leaves that client run's alone.
"""

import json
import math
import time
from collections import Counter
from pathlib import Path
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
#: What each client run adds to the server's counts: the workers' refused
#: commands and priced steps, and the frontend executor's refused jobs.
CLIENT_RUNS = (
    (
        ["command:profile on rank 0"],
        {
            "steps": 3,
            "seconds": 0.75,
            "estimated_steps": 1,
            "treatments": Counter({"uniform routing": 3}),
        },
        [],
    ),
    (
        ["command:profile on rank 0", "cost:no rung answered"],
        {
            "steps": 5,
            "seconds": 1.25,
            "analytical_steps": 2,
            "analytical_seconds": 0.5,
            "treatments": Counter(),
        },
        ["executor:render"],
    ),
)


def _serve(monkeypatch, out: Path, client_runs, served) -> Path:
    """Serve one client run per entry of `served` on one server whose run file
    declares `client_runs`; each client run sends one request, answered on
    the stream. Returns the out_dir."""
    out.mkdir()
    run = {
        "admission_path": "serving",
        "ipc_s": 2.0**-10,
        "stream_s": 2.0**-9,
        "bound_s": 100.0,
        "clock_endpoint": f"inproc:{out}",
        "out_dir": str(out),
    }
    if client_runs is not None:
        run["client_runs"] = client_runs
    (out / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(out / "run.json"))
    for module, name in [
        (compass_run, "_authority"),
        (compass_run, "_member"),
        (clock, "_installed"),
        (utils, "LP_OF_RANK"),
    ]:
        monkeypatch.setattr(module, name, None)
    table = compass_run.channel_table(run)
    with compass_run.frontend(_deployment()):
        pass
    frontend = clock.installed()
    frontend.loop = SimpleNamespace(executor=SimpleNamespace(refusals=[]))
    with compass_run.engine(_deployment()):
        pass
    engine = clock.installed()
    workers = {
        "refused_commands": [],
        "priced_steps": {"treatments": Counter()},
        "predicted_memory": {},
    }
    core = SimpleNamespace(
        runner_mgr=SimpleNamespace(call_func=lambda name, wait_out: workers[name])
    )
    with compass_run.engine_loop(core):
        frontend.ask_next_event(INF)
        engine.ask_next_event(INF, 50.0)
        for k, (commands, priced, jobs) in enumerate(served):
            workers["refused_commands"] = workers["refused_commands"] + commands
            counts = workers["priced_steps"]
            workers["priced_steps"] = {
                c: counts.get(c, 0) + priced.get(c, 0) for c in {*counts, *priced}
            }
            frontend.loop.executor.refusals += jobs
            traffic = LPRuntime(
                compass_run.TRAFFIC,
                table,
                connect(compass_run.TRAFFIC, run["clock_endpoint"]),
            )
            traffic.start_run()
            traffic.stamp_send(HTTP)
            traffic.ask_next_event(INF)
            frontend.take_grant()
            frontend.stamp_send(STREAM)
            frontend.ask_next_event(INF)
            traffic.take_grant()
            traffic.ask_next_event(INF)
            # The end of the client run: the client resigns, and the engine
            # and the frontend hand in at their marked grants, or at the finish.
            assert traffic.take_grant() == INF
            last = k + 1 == len(served)
            assert (
                (engine.take_grant() == INF) is (frontend.take_grant() == INF) is last
            )
            if not last:
                engine.ask_next_event(INF, 50.0)
                frontend.ask_next_event(INF)
    monkeypatch.setattr(clock, "_installed", frontend)
    assert compass_run.frontend_done(SimpleNamespace(close=lambda: None))
    # An earlier client run's summary is written on a thread of its own.
    deadline = time.monotonic() + 10
    for k in range(len(served)):
        while not (out / f"client-{k}" / compass_run.SUMMARY_FILE).exists():
            assert time.monotonic() < deadline, f"no client-{k} summary in 10 s"
            time.sleep(0.05)
    return out


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


@pytest.fixture
def served(monkeypatch, tmp_path):
    return _serve(monkeypatch, tmp_path / "server", 2, CLIENT_RUNS)


def test_each_client_run_has_its_own_summary_over_its_own_steps(served):
    out = served
    whole = _read(out / compass_run.SUMMARY_FILE)
    runs = [_read(out / f"client-{k}" / compass_run.SUMMARY_FILE) for k in (0, 1)]
    for k, ((commands, priced, jobs), summary) in enumerate(zip(CLIENT_RUNS, runs)):
        schedule = summary["schedule"]
        print(f"\nclient run {k}:", json.dumps(summary["client_run"]), schedule)
        assert summary["client_run"]["index"] == k
        assert [r for r, n in schedule["refusals"]["reasons"] for _ in range(n)] == (
            sorted(commands + jobs)
        )
        assert schedule["refusals"]["steps"] == priced["steps"]
        assert schedule["predicted_seconds"] == priced["seconds"]
        assert schedule["treatments"] == dict(priced["treatments"])
        by_method = schedule["seconds_by_method"]
        assert by_method["analytical"]["steps"] == priced.get("analytical_steps", 0)
        assert by_method["estimated"]["steps"] == priced.get("estimated_steps", 0)
        assert summary["coverage_report"] is True
    # The server's summary counts both client runs.
    assert whole["schedule"]["refusals"]["count"] == 4
    assert whole["schedule"]["predicted_seconds"] == 2.0
    assert runs[1]["client_run"]["start"] == runs[0]["client_run"]["end"] > 0
    assert (out / "client-0" / compass_run.LP_FILE.format("engine")).exists()


def test_a_later_client_run_s_summary_is_a_fresh_server_s(
    monkeypatch, tmp_path, served
):
    later = _read(served / "client-1" / compass_run.SUMMARY_FILE)
    fresh_out = _serve(monkeypatch, tmp_path / "fresh", None, CLIENT_RUNS[1:])
    fresh = _read(fresh_out / compass_run.SUMMARY_FILE)
    assert _read(fresh_out / "client-0" / compass_run.SUMMARY_FILE)["client_run"] == {
        "index": 0,
        "start": 0.0,
        "end": fresh["cost"]["simulated_seconds"],
    }
    for key in ("predicted_seconds", "refusals", "seconds_by_method", "treatments"):
        assert later["schedule"][key] == fresh["schedule"][key], key
    for key in ("coverage_report", "configuration", "schema_version"):
        assert later[key] == fresh[key], key
    span = later["client_run"]["end"] - later["client_run"]["start"]
    assert span == pytest.approx(fresh["cost"]["simulated_seconds"], rel=0, abs=1e-12)


def test_client_runs_is_a_positive_count_or_null(monkeypatch, tmp_path):
    for count in (0, 1.5, "2"):
        (tmp_path / "run.json").write_text(
            json.dumps({"bound_s": 1.0, "client_runs": count})
        )
        monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
        with pytest.raises(ValueError, match=f"client_runs is {count!r}"):
            compass_run.spec()


def test_a_stop_after_a_client_run_leaves_that_client_runs_summary(
    monkeypatch, tmp_path
):
    """A standalone authority stopped after client run 0 grants every server
    LP ``+inf`` and the resigned client nothing, then writes the run summary
    from every hand-in and client run 0's summary not again."""
    run = {"admission_path": "serving", "ipc_s": 2.0**-10, "stream_s": 2.0**-9}
    run.update(bound_s=100.0, client_runs=None, out_dir=str(tmp_path))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    authority = compass_run._RecordingAuthority(run)
    for lp in (compass_run.FRONTEND, compass_run.ENGINE, compass_run.TRAFFIC):
        authority.on_request(lp, "NER", INF, [])

    def hand_in(k, steps, *reasons):
        priced = {"steps": steps, "seconds": steps / 4}
        compass_run._hand_in(run, "engine", [], k, priced=priced)
        compass_run._hand_in(run, "frontend", list(reasons), k)

    hand_in(0, 1)
    client_run = tmp_path / "client-0" / compass_run.SUMMARY_FILE
    deadline = time.monotonic() + 10
    while not client_run.exists():
        assert time.monotonic() < deadline, "no client-0 summary in 10 s"
        time.sleep(0.05)
    written = client_run.read_text()
    stop = authority.on_request(compass_run.FRONTEND, "TAR", INF, [])
    assert sorted(lp.name for lp, g, _ in stop if g == INF) == ["engine", "frontend"]
    hand_in(None, 3, "ended:frontend stopped by SIGTERM")
    compass_run._write_summary(run, authority)
    summary = _read(tmp_path / compass_run.SUMMARY_FILE)
    assert summary["schedule"]["predicted_seconds"] == 0.75
    assert client_run.read_text() == written
