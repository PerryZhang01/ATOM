# SPDX-License-Identifier: MIT
"""One request across a simulated prefill-decode deployment, through atomesh.

`scripts/compass/pd_sim.sh` starts the run: the standalone clock authority, a
prefill and a decode API server on the simulated KV connector, and atomesh in
PD mode between them. The traffic LP is the vertical slice's `Traffic`, in
this process: it sends through the router and scrapes the prefill server,
since the router serves no metrics. The run is made twice with different
`PYTHONHASHSEED`, and the two step tables the authority writes are compared
byte for byte.

The run tests skip as the vertical slice's do, and without `atomesh` on PATH.
With `ATOM_COMPASS_PD_DECODE_EXEC` set, such as ``docker exec <container>``,
the decode server runs through it, in a second container on this node.

Without a driver: a frontend answers a request from outside the run while its
grant is out, both ends of the relay stamp, the standalone authority, and an
idle decode engine that a finished transfer leaves with a request to run.
"""

import asyncio
import http.client
import json
import math
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import uvicorn
from aiter_stub import stubbed_aiter
from conftest import atom_config_double
from starlette.responses import JSONResponse
from test_vertical_slice import MODEL, TREE, WALL_S, Traffic, _free_port, _run_file

from atom.compass import clock_transport
from atom.compass import run as compass_run
from atom.compass.clock import (
    NER,
    ClockAuthority,
    LpId,
    prefill_decode_table,
    single_engine_table,
)
from atom.compass.detect.determinism import compare_step_tables
from atom.utils import clock
from atom.utils.clock import LPRuntime
from atom.utils.compass_loop import HttpChannel

NEEDS_A_PD_RUN = pytest.mark.skipif(
    not (torch.cuda.is_available() and MODEL and shutil.which("atomesh")),
    reason="a run needs a driver and ATOM_COMPASS_SLICE_MODEL, as the vertical "
    "slice does, and atomesh on PATH",
)

LPS = ["engine-D", "engine-P", "frontend-D", "frontend-P", "traffic"]
RELAY = "frontend-P->frontend-D:relay"
KV_WRITE_REQ = "engine-D->engine-P:kv_write_req"
STREAM = "frontend-D->traffic:stream"
PD = {"router_s": 2.0**-12, "kv_write_req_s": 2.0**-10, "kv_link": "intra_node"}
PORTS = (
    "CLOCK_PORT",
    "PREFILL_PORT",
    "DECODE_PORT",
    "ROUTER_PORT",
    "KV_WRITE_REQ_PORT",
    "PROMETHEUS_PORT",
)
INF = math.inf


class PdTraffic(Traffic):
    http, stream = "traffic->frontend-P:http", STREAM


def run_pd(out: Path, seed: str) -> dict:
    """One run under `pd_sim.sh`; its step table and request times."""
    out.mkdir()
    ports = {name: _free_port() for name in PORTS}
    endpoint = f"tcp://127.0.0.1:{ports['CLOCK_PORT']}"
    run_file = _run_file(out, clock_endpoint=endpoint, **PD)
    env = dict(
        os.environ,
        PYTHONHASHSEED=seed,
        PYTHONPATH=str(TREE),
        AITER_LOG_LEVEL="WARNING",
        MODEL=MODEL,
        DECODE_EXEC=os.environ.get("ATOM_COMPASS_PD_DECODE_EXEC", ""),
        **{name: str(port) for name, port in ports.items()},
    )
    log = out / "pd_sim.log"
    with open(log, "w") as f:
        sim = subprocess.Popen(
            ["bash", str(TREE / "scripts/compass/pd_sim.sh"), str(run_file)],
            cwd=TREE,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    # A run that hangs ends here, with every process pd_sim.sh started.
    watchdog = threading.Timer(WALL_S, os.killpg, (sim.pid, signal.SIGKILL))
    watchdog.start()
    try:
        deadline = time.monotonic() + WALL_S
        while "pd_sim: ready" not in log.read_text():
            assert sim.poll() is None, f"pd_sim.sh exited:\n{log.read_text()}"
            assert time.monotonic() < deadline, f"no router in {WALL_S} wall seconds"
            time.sleep(0.5)
        traffic = PdTraffic(
            json.loads(run_file.read_text()),
            ports["ROUTER_PORT"],
            sim,
            scrape_port=ports["PREFILL_PORT"],
        )
        traffic.run()
        code = sim.wait(timeout=120)
    finally:
        watchdog.cancel()
        if sim.poll() is None:
            os.killpg(sim.pid, signal.SIGKILL)
    assert code == 0, f"pd_sim.sh exited {code}:\n{log.read_text()[-4000:]}"
    return {
        "table": (out / compass_run.STEP_TABLE_FILE).read_text(),
        "summary": json.loads((out / compass_run.SUMMARY_FILE).read_text()),
        "handed_in": sorted(
            f.name for f in out.glob(compass_run.COMMANDS_FILE.format("*"))
        ),
        "request": (traffic.sent, traffic.events[0][0], traffic.events[-1][0]),
        "events": traffic.events,
    }


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("pd")
    return [run_pd(root / f"seed{seed}", seed) for seed in ("1", "2")]


def _rows(table: str, kind: str, channel: str) -> list[list[str]]:
    """The step table's rows of `kind` on `channel`, each split into its fields."""
    rows = (r.split() for r in table.splitlines()[1:])
    return [r for r in rows if r[3] == kind and r[4] == channel]


@NEEDS_A_PD_RUN
def test_one_request_crosses_atomesh_to_prefill_and_decode(runs):
    run = runs[0]
    arrive, first_token, leave = run["request"]
    (relay,) = _rows(run["table"], "release", RELAY)
    (write_req,) = _rows(run["table"], "release", KV_WRITE_REQ)
    print(
        f"\nrequest 0: arrive {arrive!r} relayed {relay[1]} write request "
        f"{write_req[1]} first token {first_token!r} leave {leave!r}"
    )
    print(run["table"])
    assert run["events"][-1][1] == "data: [DONE]"
    assert len(run["events"]) > 1
    assert arrive < float(relay[1]) < float(write_req[1]) < first_token <= leave
    # Every event reached the traffic LP with its stamp, through the router.
    assert len(_rows(run["table"], "release", STREAM)) == len(run["events"])
    rows = run["table"].splitlines()[1:]
    finish = [r for r in rows if " inf " in r]
    assert sorted(r.split()[0] for r in finish) == LPS
    assert rows[-len(LPS) :] == finish


@NEEDS_A_PD_RUN
def test_the_run_summary_holds_every_server_lps_refusals(runs):
    run = runs[0]
    summary = run["summary"]
    print("\nsummary:", json.dumps(summary, sort_keys=True))
    assert run["handed_in"] == [compass_run.COMMANDS_FILE.format(lp) for lp in LPS[:-1]]
    assert summary["schedule"]["grants_total"] > 0
    assert summary["schedule"]["refusals"]["count"] == 0
    assert summary["schedule"]["refusals"]["steps"] > 0
    assert summary["coverage_report"] is False
    assert 0 < summary["cost"]["wall_seconds"]


@NEEDS_A_PD_RUN
def test_two_hash_seeds_give_byte_identical_step_tables(runs):
    left, right = (r["table"] for r in runs)
    code, report = compare_step_tables(left, right, "seed 1", "seed 2")
    print(f"\n{report}")
    assert code == 0, report
    assert left == right


# --- without a driver ---------------------------------------------------------

TABLE = single_engine_table(admission_path="serving", ipc_s=0.001, stream_s=0.002)


def test_a_frontend_answers_a_request_from_outside_the_run_while_its_grant_is_out(
    monkeypatch,
):
    """A router registering its workers asks them over HTTP, unstamped, before
    any traffic, so before any frontend can be granted time."""
    authority = clock_transport.serve(ClockAuthority(TABLE), "tcp://127.0.0.1:0")
    clock_transport.connect(LpId("engine"), authority.endpoint).send(
        (NER, INF, [], INF)
    )
    rt = LPRuntime(
        LpId("frontend"),
        TABLE,
        clock_transport.connect(LpId("frontend"), authority.endpoint),
    )
    rt.start_run()
    monkeypatch.setattr(clock, "_installed", rt)
    server = uvicorn.Server(
        uvicorn.Config(
            HttpChannel(JSONResponse({"status": "ok"}), lambda scope: None),
            host="127.0.0.1",
            port=0,
            loop="atom.utils.compass_loop:CompassEventLoop",
            lifespan="off",
            log_level="warning",
        )
    )
    got = {}

    def router():
        try:
            deadline = time.monotonic() + 10
            while not server.started and time.monotonic() < deadline:
                time.sleep(0.01)
            port = server.servers[0].sockets[0].getsockname()[1]
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            conn.request("GET", "/health")
            got["body"], got["now"] = conn.getresponse().read(), rt.now
            conn.close()
        finally:
            # Traffic joins and waits too, which finishes the run.
            traffic = clock_transport.connect(LpId("traffic"), authority.endpoint)
            traffic.send((NER, INF, [], INF))
            got["traffic"] = traffic

    thread = threading.Thread(target=router, daemon=True)
    thread.start()
    try:
        server.run()  # the finish ends it with nothing pending
    finally:
        thread.join(30)
        authority.close()
    assert got.get("body") == b'{"status":"ok"}'
    assert got["now"] == 0.0
    assert rt.now == INF


def _frontend(monkeypatch, lp: str) -> LPRuntime:
    table = prefill_decode_table(
        admission_path="serving",
        ipc_s=0.001,
        stream_s=0.002,
        router_s=PD["router_s"],
        kv_write_req_s=PD["kv_write_req_s"],
    )
    rt = LPRuntime(LpId(lp), table, None)
    monkeypatch.setattr(clock, "_installed", rt)
    return rt


async def _replay(body: bytes):
    return {"type": "http.request", "body": body, "more_body": False}


def _call(app, body: bytes = b"") -> list[dict]:
    sent = []

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/", "headers": []}
    asyncio.run(
        HttpChannel(app, lambda scope: None)(scope, lambda: _replay(body), send)
    )
    return sent


@pytest.mark.parametrize(
    "doc, stamped",
    [
        ({"kv_transfer_params": {"transfer_id": 3}}, True),
        ({"kv_transfer_params": None}, False),
        ({"status": "ok"}, False),
    ],
)
def test_a_prefill_response_with_transfer_params_is_stamped_for_the_relay(
    monkeypatch, doc, stamped
):
    rt = _frontend(monkeypatch, "frontend-P")
    start, body = _call(JSONResponse(doc))
    got = json.loads(body["body"])
    lookahead = PD["router_s"]
    if stamped:
        assert got["kv_transfer_params"]["compass"] == f"a:{lookahead!r};s:0"
        assert rt.send_log == [(RELAY, 0, lookahead)]
    else:
        assert got == doc and rt.send_log == []
    assert (b"content-length", str(len(body["body"])).encode()) in start["headers"]


def test_a_decode_request_takes_its_stamp_from_its_transfer_params(monkeypatch):
    _frontend(monkeypatch, "frontend-D")
    seen = []

    async def app(scope, receive, send):
        seen.append((await receive())["body"])
        await JSONResponse({})(scope, receive, send)

    plain = json.dumps({"kv_transfer_params": {"transfer_id": 3}}).encode()
    _call(app, plain)
    assert seen == [plain]
    stamped = json.dumps({"kv_transfer_params": {"compass": "a:0.5;s:0"}}).encode()
    # Off a CompassEventLoop, a stamped request is refused: the stamp was read.
    with pytest.raises(TypeError, match="a stamped request is received"):
        _call(app, stamped)


def test_a_prefill_decode_frontend_joins_the_standalone_authority(
    tmp_path, monkeypatch
):
    run = _run_file(tmp_path, **PD)
    authority = clock_transport.serve(
        ClockAuthority(compass_run.channel_table(json.loads(run.read_text()))),
        "tcp://127.0.0.1:0",
    )
    monkeypatch.setenv(compass_run.ENV, str(run))
    monkeypatch.setenv(compass_run.CLOCK_ENV, authority.endpoint)
    monkeypatch.setattr(clock, "_installed", None)
    monkeypatch.setattr(compass_run, "_authority", None)
    config = atom_config_double(
        kv_transfer_config={"kv_connector": "compass", "kv_role": "kv_consumer"},
        runner_qualname=compass_run.ATOM_RUNNER,
    )
    try:
        with compass_run.frontend(config):
            assert clock.installed().me == LpId("frontend-D")
        assert clock.installed().in_run
        assert compass_run._authority is None
    finally:
        authority.close()


def test_the_standalone_authority_writes_the_step_table_at_the_finish_and_the_summary_once_every_server_lp_handed_in(
    tmp_path, monkeypatch
):
    run = _run_file(tmp_path, **PD)
    # What a previous run into the same out_dir left there.
    for name in [compass_run.COMMANDS_FILE.format(lp) for lp in LPS[:-1]] + [
        compass_run.STEP_TABLE_FILE,
        compass_run.SUMMARY_FILE,
    ]:
        (tmp_path / name).write_text('["command:from the last run"]')
    monkeypatch.setenv(compass_run.ENV, str(run))
    monkeypatch.setattr(compass_run, "_authority", None)
    endpoint = f"tcp://127.0.0.1:{_free_port()}"
    serving = threading.Thread(
        target=compass_run.authority, args=(endpoint,), daemon=True
    )
    serving.start()
    deadline, conns = time.monotonic() + 10, []
    for lp in LPS:
        while True:
            try:
                conns.append(clock_transport.connect(LpId(lp), endpoint))
                break
            except ConnectionRefusedError:
                assert time.monotonic() < deadline, "no authority in 10 seconds"
                time.sleep(0.05)
    for conn in conns:
        conn.send((NER, INF, [], INF))
    assert [conn.recv()[0] for conn in conns] == [INF] * len(LPS)
    table_file = tmp_path / compass_run.STEP_TABLE_FILE
    while not table_file.exists():
        assert time.monotonic() < deadline + 10, "no step table in 10 seconds"
        time.sleep(0.05)
    rows = table_file.read_text().splitlines()[1:]
    assert sorted(r.split()[0] for r in rows) == LPS
    assert all(r.split()[2] == "inf" for r in rows)
    for conn in conns:
        conn.close()

    # Each server process hands in one refusal of each kind it holds.
    table = compass_run.channel_table(json.loads(run.read_text()))
    expected = []
    for lp in LPS[:-1]:
        rt = LPRuntime(LpId(lp), table, None)
        rt.now = INF
        rt.refusals.append(f"clock:advance_to on {lp}")
        monkeypatch.setattr(clock, "_installed", rt)
        if lp.startswith("engine"):
            refused = [f"command:reset on {lp}"]
            worker = SimpleNamespace(call_func=lambda name, wait_out, r=refused: r)
            compass_run.engine_done(SimpleNamespace(runner_mgr=worker))
        else:
            assert not (tmp_path / compass_run.SUMMARY_FILE).exists()
            refused = [f"executor:detok on {lp}"]
            rt.loop = SimpleNamespace(executor=SimpleNamespace(refusals=refused))
            assert compass_run.frontend_done(SimpleNamespace(close=lambda: None))
        expected += [[refused[0], 1], [rt.refusals[0], 1]]
    serving.join(10)
    assert not serving.is_alive()
    summary = json.loads((tmp_path / compass_run.SUMMARY_FILE).read_text())
    print("refusals:", json.dumps(summary["schedule"]["refusals"]))
    assert summary["schedule"]["refusals"]["reasons"] == sorted(expected)
    assert summary["coverage_report"] is True


def test_a_summary_missing_a_server_lps_refusals_names_it_though_the_last_run_left_one(
    tmp_path, monkeypatch
):
    run = json.loads(_run_file(tmp_path, **PD).read_text())
    monkeypatch.setattr(compass_run, "HAND_IN_WAIT_S", 0.2)
    (tmp_path / compass_run.COMMANDS_FILE.format("engine-D")).write_text("[]")
    authority = compass_run._RecordingAuthority(run)
    (tmp_path / compass_run.COMMANDS_FILE.format("engine-P")).write_text("[]")
    with pytest.raises(TimeoutError, match="no commands-engine-D.json, commands-fr"):
        compass_run._write_summary(run, authority)


def test_an_idle_engine_runs_at_once_a_request_a_finished_transfer_left_ready(
    monkeypatch,
):
    """Decode's idle pass polls its transfers after `schedule()`, so the request
    a finished transfer leaves ready runs on the next pass, with no time between."""
    with stubbed_aiter():
        from atom.model_engine.engine_core import KV_IDLE_DRAIN_INTERVAL_S, EngineCore
    rt = LPRuntime(LpId("engine"), TABLE, None)
    rt.now = 2.0
    monkeypatch.setattr(clock, "_installed", rt)
    engine = EngineCore.__new__(EngineCore)
    engine.kv_transfer_enabled = True
    connector = SimpleNamespace(has_pending_work=lambda: True)
    engine.scheduler = SimpleNamespace(
        waiting=[SimpleNamespace(id=7)],
        finished_recving_kv_req_ids=[],
        failed_recving_kv_req_ids=[],
        deferred_free_blocks={},
        kv_connector=connector,
    )
    assert engine._idle_deadline() == 2.0 + KV_IDLE_DRAIN_INTERVAL_S
    connector.has_pending_work = lambda: False
    engine.scheduler.finished_recving_kv_req_ids.append(8)  # no such request waits
    assert engine._idle_deadline() == INF
    engine.scheduler.finished_recving_kv_req_ids.append(7)
    assert engine._idle_deadline() == 2.0
