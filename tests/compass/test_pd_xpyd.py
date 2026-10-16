# SPDX-License-Identifier: MIT
"""Two prefill and two decode deployments behind the router LP (2P2D).

Each deployment is its own frontend and engine LP pair, the router chooses
among them on its own clock, and decode sends each write request to the
prefill deployment the relayed engine id names.

Without a driver: the channel table and the refusals; the connector's choice
of deployment; atomesh, built from this tree, between scripted frontends as in
`test_router_lp_acceptance`, twice, under two hash seeds and random host delays.
With one, `pd_sim.sh` runs 2P2D at TP1 under a constant stub, twice, and
`check_run` holds the runs to the scenario checks and each engine LP's rows to
the timeline the stub implies.
"""

import json
import os
import random
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from collections import Counter
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_dp_slice import MAX_TOKENS
from test_kv_remote_prefill import Script, connector
from test_kv_simulated_connector import PEAKS, geometry, model_for  # noqa: F401
from test_pd_dp import RouterLpRequests
from test_pd_slice import NEEDS_A_PD_RUN, PD
from test_router_lp_acceptance import (
    WALL_S,
    DelayedTraffic,
    Frontend,
    U,
    _idle,
    _launch,
    atomesh,  # noqa: F401
)
from test_vertical_slice import CONSTANT, MODEL, TREE, _free_port, _retry, _run_file

from atom.compass import run as compass_run
from atom.compass.clock import LpId, prefill_decode_table
from atom.compass.detect.scenario import check_run
from atom.kv_transfer.disaggregation.types import KVConnectorOutput
from atom.utils.clock import LPRuntime

XPYD = {"prefill_instances": 2, "decode_instances": 2}
ROUTER_IN_S = 2.0**-14


def _table(prefill=2, decode=2, **kw):
    return prefill_decode_table(
        admission_path="serving",
        ipc_s=ROUTER_IN_S,
        stream_s=PD["router_s"],
        router_s=PD["router_s"],
        kv_write_req_s=PD["kv_write_req_s"],
        prefill=prefill,
        decode=decode,
        **kw,
    )


def test_each_deployment_is_a_frontend_and_engine_lp_with_its_own_channels():
    table = _table(router_in_s=ROUTER_IN_S)
    names = {c.name for lp in table.registry.ids() for c in table.channels_from(lp)}
    assert sorted(lp.name for lp in table.registry.ids()) == sorted(
        ["traffic", "router"]
        + [f"{r}-{s}{k}" for r in ("frontend", "engine") for s in "PD" for k in (0, 1)]
    )
    assert {"traffic->router:http", "router->traffic:stream"} <= names
    for k in (0, 1):
        assert {
            f"router->frontend-P{k}:http",
            f"frontend-P{k}->router:relay",
            f"router->frontend-D{k}:relay",
            f"frontend-D{k}->router:stream",
        } <= names
    assert sorted(n for n in names if ":kv_write_req" in n) == [
        f"engine-D{d}->engine-P{p}:kv_write_req" for d in (0, 1) for p in (0, 1)
    ]


def test_one_deployment_per_side_keeps_its_names():
    table = _table(1, 1, router_in_s=ROUTER_IN_S)
    assert {lp.name for lp in table.registry.ids()} == {
        "traffic",
        "router",
        "frontend-P",
        "engine-P",
        "frontend-D",
        "engine-D",
    }


def test_several_deployments_without_the_router_lp_are_refused_by_name():
    with pytest.raises(ValueError, match="2 prefill and 1 decode .* set router_in_s"):
        _table(2, 1)


def _server(run: dict, role: str, **kv):
    kv = {"kv_connector": "compass", "kv_role": role} | kv
    return SimpleNamespace(kv_transfer_config=kv)


def test_a_server_runs_as_the_deployment_its_index_names():
    name = compass_run._pd_name
    assert name(_server(XPYD, "kv_producer", compass_instance=1), XPYD) == "P1"
    assert name(_server(XPYD, "kv_consumer", compass_instance=0), XPYD) == "D0"
    assert name(_server({}, "kv_producer"), {}) == "P"
    for kv in ({}, {"compass_instance": 2}, {"compass_instance": "1"}):
        with pytest.raises(ValueError, match="compass_instance is .* declares 2"):
            name(_server(XPYD, "kv_producer", **kv), XPYD)


def test_a_replay_with_several_deployments_of_a_side_is_refused_by_name(
    tmp_path, monkeypatch
):
    path = _run_file(tmp_path, replay=str(tmp_path), **XPYD)
    run = json.loads(path.read_text())
    del run["step_stub"]
    path.write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(path))
    config = SimpleNamespace(
        tensor_parallel_size=1, parallel_config=SimpleNamespace(data_parallel_rank=0)
    )
    with pytest.raises(ValueError, match="several deployments of a side"):
        compass_run.runner(SimpleNamespace(config=config), None)


# --- the connector -----------------------------------------------------------


def test_decode_sends_the_write_request_to_the_prefill_deployment_that_served_it(
    geometry, monkeypatch  # noqa: F811
):
    """Prefill deployment 1 finishes a request; its relayed blob reaches decode
    deployment 0, whose write request goes to deployment 1 only."""
    table = _table(router_in_s=ROUTER_IN_S)
    model = model_for(geometry, PEAKS[0])
    address = f"inproc://kv-write-req-{uuid.uuid4().hex}:5000"

    def half(lp, role):
        rt = LPRuntime(LpId(lp), table, Script())
        rt.start_run()
        return rt, connector(rt, monkeypatch, model, kv_role=role, address=address)

    (_, p0), (rt_p1, p1) = half("engine-P0", "kv_producer"), half(
        "engine-P1", "kv_producer"
    )
    rt_d, d = half("engine-D0", "kv_consumer")
    seq = SimpleNamespace(
        id=9,
        kv_transfer_params={"do_remote_decode": True},
        block_table=[0, 1],
        output_tokens=[7],
    )
    p1.request_finished(seq)
    blob = seq.kv_transfer_params_output
    assert blob["remote_engine_id"] == "compass-simulated-p1"
    assert blob["remote_host"] == "compass-simulated-p1.invalid"
    d.update_state_after_alloc(
        SimpleNamespace(
            id=9,
            kv_transfer_params=blob,
            block_table=[0, 1],
            has_per_req_cache=False,
            num_cached_tokens=0,
        )
    )
    channel, seq_no, arrival = rt_d.send_log[-1]
    assert channel == "engine-D0->engine-P1:kv_write_req"
    rt_p1.conn.grants.append((1.0, {channel: [(seq_no, arrival)]}))
    rt_p1.advance_to(1.0)
    p1.process_completions(KVConnectorOutput())
    assert p1.process_completions(KVConnectorOutput()).finished_sending == {9}
    assert not p0._socks[("engine-D0", 0)].raw.poll(0)


# --- atomesh between scripted frontends ---------------------------------------

#: Eight requests a sixteenth of U apart, two tokens each: all are in prefill
#: at once, so each side's least-load choice walks its workers in the order
#: the router sorts them, base URL then DP rank, and starts again.
WORKLOAD = [(k * U / 16, 2) for k in range(8)]
#: Request k's (deployment, DP rank) on each side.
PLACED = [((k % 4) // 2, k % 2) for k in range(8)]


class InstanceFrontend(Frontend):
    """`Frontend` as LP `lp`, serving on `http`, which nothing serves yet."""

    def __init__(self, run: dict, side: str, delay, lp: str, http) -> None:
        self.run, self.side, self.delay, self.lp = run, side, delay, LpId(lp)
        self.cv = threading.Condition()
        self.bodies, self.replies, self.ranks, self.errors = {}, {}, {}, []
        self.http, self.url = http, f"http://127.0.0.1:{http.server_port}"
        http.RequestHandlerClass = self._handler()
        threading.Thread(target=http.serve_forever, daemon=True).start()
        self.owner = threading.Thread(target=self._own, daemon=True)
        self.owner.start()


def run_scripted_xpyd(binary: Path, out: Path, hash_seed: str, delay) -> list:
    """One run in `out`; each request's (prefill, decode) (deployment, rank)."""
    out.mkdir()
    run_file = _run_file(
        out,
        data_parallel_size=2,
        step_stub=CONSTANT,
        router_in_s=ROUTER_IN_S,
        **XPYD,
        **PD,
    )
    run = json.loads(run_file.read_text())
    table = compass_run.channel_table(run)
    lookaheads = {
        c.name: table.lookahead(c.name) for c in table.channels_from(compass_run.ROUTER)
    }
    endpoint = run["clock_endpoint"]
    errors, procs, frontends = [], [], {}
    try:
        procs.append(
            authority := _launch(
                out,
                "authority",
                [sys.executable, "-m", "atom.compass.run", "--compass-run"]
                + [str(run_file), "--compass-clock-endpoint", endpoint],
                PYTHONHASHSEED=hash_seed,
                PYTHONPATH=str(TREE),
            )
        )
        host, port = endpoint.removeprefix("tcp://").rsplit(":", 1)
        _retry(
            "clock", authority, lambda: socket.create_connection((host, port)).close()
        )
        # Deployment k of a side serves on that side's k-th URL in sorted order,
        # the order the router takes its workers in.
        for side in "PD":
            servers = sorted(
                (ThreadingHTTPServer(("127.0.0.1", 0), None) for _ in range(2)),
                key=lambda s: f"http://127.0.0.1:{s.server_port}",
            )
            for k, http in enumerate(servers):
                lp = f"frontend-{side}{k}"
                frontends[lp] = InstanceFrontend(run, side, delay, lp, http)
        idle = [
            threading.Thread(
                target=_idle, args=(run, f"engine-{s}{k}", f"dp{r}", errors)
            )
            for s in "PD"
            for k in range(2)
            for r in range(2)
        ]
        for t in idle:
            t.daemon = True
            t.start()
        router_port = _free_port()
        urls = [
            arg
            for lp, f in frontends.items()
            for arg in ("--prefill" if "-P" in lp else "--decode", f.url)
        ]
        procs.append(
            router := _launch(
                out,
                "router",
                [str(binary), "launch", "--host", "127.0.0.1", "--port"]
                + [str(router_port), "--pd-disaggregation", *urls]
                + ["--policy", "dp_sticky", "--dp-aware", "--backend", "atom"]
                + ["--disable-health-check", "--disable-circuit-breaker"]
                + ["--disable-retries", "--prometheus-port", str(_free_port())]
                + ["--log-level", "info"],
                ATOM_COMPASS_CLOCK_ENDPOINT=endpoint,
                ATOM_COMPASS_SEED="20260707",
                ATOM_COMPASS_ROUTER_LOOKAHEADS=json.dumps(lookaheads),
                ATOM_COMPASS_ROUTER_WORKERS=json.dumps(
                    {f.url: lp for lp, f in frontends.items()}
                ),
            )
        )
        deadline = time.monotonic() + WALL_S
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{router_port}/v1/models")
                break
            except OSError:
                assert router.poll() is None, (out / "router.log").read_text()
                assert time.monotonic() < deadline, f"no router in {WALL_S} s"
                time.sleep(0.5)
        bodies = [
            {"model": "m", "prompt": f"request {i}", "max_tokens": n, "stream": True}
            for i, (_, n) in enumerate(WORKLOAD)
        ]
        offsets = [t for t, _ in WORKLOAD]

        def drive() -> None:
            try:
                DelayedTraffic(run, router_port, bodies, offsets, delay=delay).run()
            except BaseException as e:  # noqa: BLE001 - the test raises it
                errors.append(e)

        # A message the router sends to the wrong frontend never arrives, and
        # the LPs wait for it: one wall deadline bounds them all.
        lps = [threading.Thread(target=drive, daemon=True)]
        lps[0].start()
        lps += [*idle, *(f.owner for f in frontends.values())]
        deadline = time.monotonic() + WALL_S
        for t in lps:
            t.join(max(0.0, deadline - time.monotonic()))
        stuck = [t.name for t in lps if t.is_alive()]
        assert not stuck, f"still running after {WALL_S} s: {stuck}"
        assert authority.wait(WALL_S) == 0, (out / "authority.log").read_text()
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
        for f in frontends.values():
            f.http.shutdown()
    errors += [e for f in frontends.values() for e in f.errors]
    if errors:
        raise errors[0]
    placed = {}
    for lp, f in frontends.items():
        for request, rank in f.ranks.items():
            placed.setdefault(request, {})[lp[len("frontend-")]] = (int(lp[-1]), rank)
    return [(placed[i]["P"], placed[i]["D"]) for i in range(len(WORKLOAD))]


@pytest.fixture(scope="module")
def scripted_runs(atomesh, tmp_path_factory):  # noqa: F811
    root = tmp_path_factory.mktemp("xpyd_scripted")
    wall = random.Random(2)
    delays = {"1": lambda: 0.0, "2": lambda: wall.uniform(0.0, 0.02)}
    return {
        root / f"seed{s}": run_scripted_xpyd(atomesh, root / f"seed{s}", s, delay)
        for s, delay in delays.items()
    }


def test_2p2d_through_atomesh_meets_every_scenario_check(scripted_runs):
    failed = check_run(list(scripted_runs))
    assert not failed, "\n".join(failed)


def test_each_request_takes_the_least_load_deployment_and_rank_under_any_delay(
    scripted_runs,
):
    want = [(p, p) for p in PLACED]
    print("\n(prefill, decode) (deployment, rank) per request:", scripted_runs)
    assert list(scripted_runs.values()) == [want, want]


# --- with a driver: pd_sim.sh at 2P2D ------------------------------------------

#: Each port name pd_sim.sh reads, by its offset in a block of free ports; a
#: server k of a side takes its port plus k and its DP port plus 16k.
OFFSETS = {
    "CLOCK_PORT": 0,
    "ROUTER_PORT": 1,
    "PROMETHEUS_PORT": 2,
    "PREFILL_PORT": 3,
    "DECODE_PORT": 5,
    "KV_WRITE_REQ_PORT": 7,
    "PREFILL_DP_PORT": 11,
    "DECODE_DP_PORT": 12,
}


def _free_block(n: int) -> int:
    """The first of `n` consecutive ports nothing listens on."""
    while True:
        base, free = _free_port(), True
        for p in range(base, base + n):
            with socket.socket() as s:
                free = free and s.connect_ex(("127.0.0.1", p)) != 0
        if free and base + n < 65536:
            return base


def run_xpyd(out: Path, seed: str) -> dict:
    """One 2P2D run under `pd_sim.sh`, round robin on each side."""
    out.mkdir()
    base = _free_block(32)
    ports = {name: base + k for name, k in OFFSETS.items()}
    run_file = _run_file(
        out,
        clock_endpoint=f"tcp://127.0.0.1:{ports['CLOCK_PORT']}",
        step_stub=CONSTANT,
        **XPYD,
        **PD,
    )
    env = dict(
        os.environ,
        PYTHONHASHSEED=seed,
        PYTHONPATH=str(TREE),
        AITER_LOG_LEVEL="WARNING",
        ATOM_COMPASS_PARITY_RECORD=str(out / "parity"),
        MODEL=MODEL,
        ROUTER_POLICY="round_robin",
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
    watchdog = threading.Timer(WALL_S * 2, os.killpg, (sim.pid, signal.SIGKILL))
    watchdog.start()
    try:
        deadline = time.monotonic() + WALL_S * 2
        while "pd_sim: ready" not in log.read_text():
            assert sim.poll() is None, f"pd_sim.sh exited:\n{log.read_text()}"
            assert time.monotonic() < deadline, "no router in time"
            time.sleep(0.5)
        run = json.loads(run_file.read_text())
        traffic = RouterLpRequests(run, ports["ROUTER_PORT"], sim)
        traffic.run()
        code = sim.wait(timeout=120)
    finally:
        watchdog.cancel()
        if sim.poll() is None:
            os.killpg(sim.pid, signal.SIGKILL)
    assert code == 0, f"pd_sim.sh exited {code}:\n{log.read_text()[-4000:]}"
    return {
        "out": out,
        "table": (out / compass_run.STEP_TABLE_FILE).read_text(),
        "events": traffic.events,
    }


@pytest.fixture(scope="module")
def xpyd_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("xpyd")
    return [run_xpyd(root / f"seed{s}", s) for s in ("1", "2")]


@NEEDS_A_PD_RUN
def test_2p2d_meets_every_scenario_check_with_requests_on_every_deployment(
    xpyd_runs,
):
    run = xpyd_runs[0]
    rows = [r.split() for r in run["table"].splitlines()[1:]]
    pairs = Counter(
        r[4].partition(":")[0]
        for r in rows
        if r[3] == "release" and ":kv_write_req" in r[4]
    )
    print(
        "\nKV transfers per decode->prefill engine pair:", dict(sorted(pairs.items()))
    )
    parity = run["out"] / "parity"
    records = {
        f"engine-{s}{k}": parity / f"{n}{k}"
        for s, n in (("P", "prefill"), ("D", "decode"))
        for k in (0, 1)
    }
    failed = check_run(
        [r["out"] for r in xpyd_runs],
        MAX_TOKENS,
        [line for _, line in run["events"]],
        records,
    )
    assert not failed, "\n".join(failed)
    assert sum(pairs.values()) == len(MAX_TOKENS)
    ends = [p.split("->") for p in pairs]
    assert {d for d, _ in ends} == {"engine-D0", "engine-D1"}
    assert {p for _, p in ends} == {"engine-P0", "engine-P1"}
